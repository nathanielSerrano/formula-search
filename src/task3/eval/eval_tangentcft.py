"""
ARQMath Task 2 evaluation: formula-to-formula retrieval on ARQMath-3 (Year 3) 
Modified for TangentCFT Universal Evaluation (OPT, SLT, or Dual Fusion).
Features SQLite Disk-Caching and Progressive FAISS Search to prevent OOM Kills.
"""

from __future__ import annotations

import argparse
import json
import gc
import re
import sys
import os
import sqlite3
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytrec_eval
from tqdm import tqdm
from gensim.models import FastText

_PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(_PROJECT_ROOT))

_TANGENT_CFT_ROOT = (_PROJECT_ROOT.parent / "TangentCFT").resolve()
if not _TANGENT_CFT_ROOT.exists():
    _TANGENT_CFT_ROOT = Path("/home/nathaniel.serrano/formula-search")
sys.path.insert(0, str(_TANGENT_CFT_ROOT))

from Embedding_Preprocessing.encoder_tuple_level import TupleEncoder, TupleTokenizationMode
from TangentS.math_tan.math_extractor import MathExtractor
from TangentS.math_tan.symbol_tree import SymbolTree

_FORMULA_INDEX_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_EVAL_SPLIT = "eval"
_QUICK_RUN_CORPUS_SIZE = 100_000
_BATCH_SIZE = 250_000 # Creates a lightweight ~600MB FAISS index at a time

# ---------------------------------------------------------------------------
# Qrels and Topics Loaders
# ---------------------------------------------------------------------------

_QREL_PATHS: Dict[str, List[Path]] = {
    "eval": [_PROJECT_ROOT / "data/raw/arqmath/qrels/task2/arqmath3/qrel_task2_2022_official.tsv"],
}
_TOPIC_PATHS: Dict[str, List[Path]] = {
    "eval": [_PROJECT_ROOT / "data/raw/arqmath/topics/task2/arqmath3/Topics_Task2_2022_V0.1.xml"],
}

def load_topics(split: str) -> Dict[str, str]:
    topics: Dict[str, str] = {}
    for path in _TOPIC_PATHS[split]:
        if not path.exists(): raise FileNotFoundError(f"Topic file not found: {path}")
        tree = ET.parse(str(path))
        for topic in tree.getroot():
            num = topic.get("number", "")
            latex_elem = topic.find("Latex")
            if num and latex_elem is not None and latex_elem.text:
                topics[num] = latex_elem.text.strip()
    return topics

def load_qrels(split: str) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    for path in _QREL_PATHS[split]:
        if not path.exists(): raise FileNotFoundError(f"Qrel file not found: {path}")
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4: continue
                topic, _, cand_id, grade = parts[0], parts[1], parts[2], parts[3]
                qrels.setdefault(topic, {})[cand_id] = float(grade)
    return qrels

# ---------------------------------------------------------------------------
# TangentCFT Vectorization Logic (For Queries Only)
# ---------------------------------------------------------------------------

def extract_tuples_from_math(mathml: str, is_slt: bool = False):
    try:
        mathml = re.sub(r'<csymbol[^>]*>', '<ci>', mathml)
        mathml = mathml.replace('</csymbol>', '</ci>')
        
        if is_slt:
            pmml = MathExtractor.isolate_pmml(mathml)
            if not pmml: pmml = mathml
            temp = SymbolTree(MathExtractor.convert_to_layoutsymbol(pmml), -1, [0])
        else:
            if 'encoding="MathML-Content"' in mathml and '<annotation-xml' not in mathml:
                cmml = mathml
            else:
                cmml = MathExtractor.isolate_cmml(mathml)
            if not cmml: raise ValueError("Could not isolate CMML.")
            current_tree = MathExtractor.convert_to_semanticsymbol(cmml)
            temp = SymbolTree(current_tree)
    except Exception as e:
        return None
        
    return temp.get_pairs(window=2, eob=True)

def get_xml_embedding(
    mathml_string: str, model: FastText, node_map: dict, edge_map: dict, is_slt: bool = False
) -> np.ndarray:
    vec_dim = model.vector_size
    if not mathml_string or not isinstance(mathml_string, str): return np.zeros(vec_dim, dtype=np.float32)

    tuple_list = extract_tuples_from_math(mathml_string, is_slt=is_slt)
    if not tuple_list: return np.zeros(vec_dim, dtype=np.float32)

    try:
        encoded_tokens, _, _, _, _ = TupleEncoder.encode_tuples(
            node_map.copy(), edge_map.copy(), 1000000, 100000, tuple_list,
            TupleTokenizationMode.Both_Separated, ignore_full_relative_path=True,
            tokenize_all=False, tokenize_number=True
        )

        vectors = [model.wv[t] for t in encoded_tokens if t in model.wv]
        if not vectors: return np.zeros(vec_dim, dtype=np.float32)

        return np.mean(vectors, axis=0)
    except Exception as e:
        return np.zeros(vec_dim, dtype=np.float32)

def _get_token_vectors(data_encoded, vectors_matrix, vocab_dict, vec_size):
    valid_indices = [vocab_dict[t] for t in data_encoded if t in vocab_dict]
    if not valid_indices: return np.zeros(vec_size, dtype=np.float32)
    return np.mean(vectors_matrix[valid_indices], axis=0)

# ---------------------------------------------------------------------------
# Query Encoding 
# ---------------------------------------------------------------------------

def _encode_queries_universal(
    mode: str, base_model: FastText, base_node: dict, base_edge: dict,
    slt_model: Optional[FastText], slt_node: Optional[dict], slt_edge: Optional[dict]
) -> Dict[str, Optional[np.ndarray]]:
    
    topics = load_topics(_EVAL_SPLIT)
    qrels = load_qrels(_EVAL_SPLIT)
    
    # 1. Maintain strict ranking of proxies from the qrels file
    topic_to_ranked_proxies = {}
    all_target_vids = set()
    for topic_id in topics.keys():
        if topic_id in qrels:
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            if positives:
                topic_to_ranked_proxies[topic_id] = positives
                all_target_vids.update(positives)
                
    proxy_xmls = {}
    shard_files = sorted(list(_FORMULA_INDEX_DIR.glob("*.parquet")))
    print(f"Searching Parquet files for query XMLs...", flush=True)
    
    # 2. Gather XMLs safely without race conditions
    for shard in shard_files:
        try:
            parquet_file = pq.ParquetFile(shard)
            schema_names = parquet_file.schema.names
        except Exception: continue
            
        vid_col = "old_visual_id" if "old_visual_id" in schema_names else "visual_id"
        if vid_col not in schema_names: continue
        
        read_cols = [vid_col]
        if "opt" in schema_names: read_cols.append("opt")
        if "slt" in schema_names: read_cols.append("slt")

        df = parquet_file.read(columns=read_cols).to_pandas()
        mask = df[vid_col].astype(str).isin(all_target_vids)
        for _, row in df[mask].iterrows():
            vid_str = str(row[vid_col])
            proxy_xmls[vid_str] = {
                "opt": row["opt"] if "opt" in row and pd.notna(row["opt"]) else "",
                "slt": row["slt"] if "slt" in row and pd.notna(row["slt"]) else ""
            }
            
        del df
        gc.collect()
        if len(proxy_xmls) >= len(all_target_vids): break
            
    # 3. Deterministically choose the best proxy, prioritizing positives[0]
    query_xmls = {}
    for topic, ranked_vids in topic_to_ranked_proxies.items():
        for vid in ranked_vids:
            if vid in proxy_xmls:
                xml_data = proxy_xmls[vid]
                
                if mode == "opt" and xml_data["opt"]:
                    query_xmls[topic] = xml_data
                    break
                elif mode == "slt" and xml_data["slt"]:
                    query_xmls[topic] = xml_data
                    break
                elif mode == "fused" and xml_data["opt"] and xml_data["slt"]:
                    query_xmls[topic] = xml_data
                    break
                    
        # Fallback for Fused mode if a perfect pair doesn't exist
        if topic not in query_xmls and mode == "fused":
            for vid in ranked_vids:
                if vid in proxy_xmls and proxy_xmls[vid]["opt"]:
                    query_xmls[topic] = proxy_xmls[vid]
                    break
    
    query_embs = {}
    for topic in tqdm(topics.keys(), desc="Encoding Queries"):
        if topic not in query_xmls: continue
        xml_data = query_xmls[topic]

        if mode == "opt":
            fused = get_xml_embedding(xml_data["opt"], base_model, base_node, base_edge, is_slt=False)
        elif mode == "slt":
            fused = get_xml_embedding(xml_data["slt"], base_model, base_node, base_edge, is_slt=True)
        elif mode == "fused":
            opt_vec = get_xml_embedding(xml_data["opt"], base_model, base_node, base_edge, is_slt=False)
            slt_vec = get_xml_embedding(xml_data["slt"], slt_model, slt_node, slt_edge, is_slt=True)
            fused = np.concatenate([opt_vec, slt_vec])
            
        norm = np.linalg.norm(fused)
        if norm > 0: 
            query_embs[topic] = fused / norm

    print(f"Resolved {len(query_embs)}/{len(topics)} query topics", flush=True)
    return query_embs

# ---------------------------------------------------------------------------
# Corpus encoding (SQLite Cache + Progressive FAISS)
# ---------------------------------------------------------------------------

def _progressive_search(
    args, query_embs: Dict[str, np.ndarray], base_model: FastText, slt_model: Optional[FastText]
) -> Dict[str, Dict[str, float]]:
    
    query_topics = list(query_embs.keys())
    if not query_topics: return {}
    query_mat = np.vstack([query_embs[t] for t in query_topics]).astype(np.float32)
    
    # 1. Build SQLite Disk Cache for Fused Mode (0 RAM Overhead)
    conn = None
    slt_dim = 0
    if args.mode == "fused":
        db_path = "slt_vectors_cache.db"
        if os.path.exists(db_path): os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("CREATE TABLE slt (id TEXT PRIMARY KEY, vec BLOB)")
        
        slt_mat = slt_model.wv.vectors
        slt_vocab = slt_model.wv.key_to_index if hasattr(slt_model.wv, 'key_to_index') else {k: v.index for k, v in slt_model.wv.vocab.items()}
        slt_dim = slt_model.vector_size
        
        batch = []
        with open(args.slt_corpus_jsonl, "r", encoding="utf-8") as f:
            for idx, line in tqdm(enumerate(f), desc="Writing SLT Disk Cache"):
                if args.quick_run and idx >= _QUICK_RUN_CORPUS_SIZE: break
                data = json.loads(line)
                vec = _get_token_vectors(data["encoded"], slt_mat, slt_vocab, slt_dim)
                batch.append((str(data["id"]), vec.tobytes()))
                
                if len(batch) >= 100000:
                    conn.executemany("INSERT INTO slt VALUES (?, ?)", batch)
                    batch.clear()
        if batch:
            conn.executemany("INSERT INTO slt VALUES (?, ?)", batch)
        conn.commit()

    # 2. Stream Base Vectors & Progressive FAISS Search
    target_dim = base_model.vector_size + slt_dim if args.mode == "fused" else base_model.vector_size
    topk_results = {t: [] for t in query_topics}
    
    base_mat = base_model.wv.vectors
    base_vocab = base_model.wv.key_to_index if hasattr(base_model.wv, 'key_to_index') else {k: v.index for k, v in base_model.wv.vocab.items()}
    base_dim = base_model.vector_size
    
    batch_embs = []
    batch_ids = []
    zeros = np.zeros(slt_dim, dtype=np.float32) if args.mode == "fused" else None
    cursor = conn.cursor() if conn else None

    def search_and_flush():
        nonlocal batch_embs, batch_ids
        if not batch_embs: return
        
        mat = np.vstack(batch_embs).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.divide(mat, norms, out=np.zeros_like(mat), where=norms!=0)
        
        index = faiss.IndexFlatIP(target_dim)
        index.add(mat)
        
        scores, indices = index.search(query_mat, min(args.top_k, len(batch_embs)))
        for q_idx, topic in enumerate(query_topics):
            for rank_idx, match_idx in enumerate(indices[q_idx]):
                if match_idx >= 0:
                    topk_results[topic].append((float(scores[q_idx][rank_idx]), batch_ids[match_idx]))
                    
        batch_embs.clear()
        batch_ids.clear()
        gc.collect()

    with open(args.corpus_jsonl, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc="Progressive Search"):
            if args.quick_run and idx >= _QUICK_RUN_CORPUS_SIZE: break
            data = json.loads(line)
            vid = str(data["id"])
            b_vec = _get_token_vectors(data["encoded"], base_mat, base_vocab, base_dim)
            
            if args.mode == "fused":
                cursor.execute("SELECT vec FROM slt WHERE id=?", (vid,))
                row = cursor.fetchone()
                s_vec = np.frombuffer(row[0], dtype=np.float32) if row else zeros
                batch_embs.append(np.concatenate([b_vec, s_vec]))
            else:
                batch_embs.append(b_vec)
                
            batch_ids.append(vid)
            
            if len(batch_embs) >= _BATCH_SIZE:
                search_and_flush()
                
    search_and_flush()
    
    if conn:
        conn.close()
        if os.path.exists("slt_vectors_cache.db"): os.remove("slt_vectors_cache.db")
        
    # 3. Consolidate Results
    run = {}
    for topic, results in topk_results.items():
        results.sort(key=lambda x: x[0], reverse=True)
        run[topic] = {doc_id: score for score, doc_id in results[:args.top_k]}
        
    return run

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    print(f"Running in {args.mode.upper()} mode...", flush=True)
    
    base_model = FastText.load(args.model_path)
    node_prefix = "slt" if args.mode == "slt" else "opt"
    with open(Path(args.dict_dir) / f"{node_prefix}_node_map.json", "r") as f: base_node = json.load(f)
    with open(Path(args.dict_dir) / f"{node_prefix}_edge_map.json", "r") as f: base_edge = json.load(f)

    slt_model, slt_node, slt_edge = None, None, None
    if args.mode == "fused":
        print("Loading secondary SLT model for fusion...", flush=True)
        slt_model = FastText.load(args.slt_model_path)
        with open(Path(args.slt_dict_dir) / "slt_node_map.json", "r") as f: slt_node = json.load(f)
        with open(Path(args.slt_dict_dir) / "slt_edge_map.json", "r") as f: slt_edge = json.load(f)

    # Note the reversed order: Queries are encoded FIRST, then corpus is chunk-searched!
    query_embs = _encode_queries_universal(
        args.mode, base_model, base_node, base_edge, slt_model, slt_node, slt_edge
    )

    run = _progressive_search(args, query_embs, base_model, slt_model)

    qrels_eval = load_qrels(_EVAL_SPLIT)
    qrels_int = {t: {c: int(g) for c, g in cands.items()} for t, cands in qrels_eval.items() if t in run}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_int, {"ndcg_cut", "map_cut", "P", "bpref"}, relevance_level=2)
    results = evaluator.evaluate(run)

    metrics_agg = defaultdict(float)
    n = len(results)
    if n == 0:
        print("\nNo queries were resolved successfully. Halting evaluation.")
        return
        
    for topic_results in results.values():
        for metric, value in topic_results.items(): metrics_agg[metric] += value
    metrics_agg = {k: v / n for k, v in metrics_agg.items()}

    print(f"\n{'='*50}\nTask 3 Evaluation — {n} topics\n{'='*50}")
    if "bpref" in metrics_agg: print(f"  bpref      {metrics_agg['bpref']:.4f}\n{'-'*30}")
    for k in [5, 10, 100, 1000]:
        if f"ndcg_cut_{k}" in metrics_agg: print(f"  nDCG@{k:<5} {metrics_agg[f'ndcg_cut_{k}']:.4f}")
        if f"map_cut_{k}" in metrics_agg: print(f"  MAP@{k:<6} {metrics_agg[f'map_cut_{k}']:.4f}")
        if f"P_{k}" in metrics_agg: print(f"  P@{k:<8} {metrics_agg[f'P_{k}']:.4f}")
        if k != 1000: print(f"{'-'*30}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["opt", "slt", "fused"], default="opt")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dict-dir", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--slt-model-path", type=str, default=None)
    parser.add_argument("--slt-dict-dir", type=str, default=None)
    parser.add_argument("--slt-corpus-jsonl", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--quick-run", action="store_true")
    
    evaluate(parser.parse_args())