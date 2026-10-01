"""
Build reranker feature files: one row per (topic, candidate formula).

Candidates are either
  --judged          every judged visual id of the split's qrels (training data: every row
                    has a grade), or
  --run FILE        the top --depth results of a first-stage run (what gets reranked).

Features (src.rerank.features.FEATURE_NAMES, then):
  bm25, bm25_norm            BM25 score of the candidate for the query (raw, and min-max
                             normalised over the topic's candidates)
  gnn_pre, gnn_pre_norm      cosine similarity under the *pretrained* encoder, which never
                             saw relevance judgments; fine-tuned scores would be inflated on
                             the topics the fine-tuned model was trained on

The proto group's IDF-weighted overlaps use document frequencies from a seeded random
sample of --idf-sample corpus formulas, cached as data/processed/rerank/idf_<n>_<seed>.npz
so every feature file built with the same settings shares one table.

Output: data/processed/rerank/<name>.npz with topics, visual_ids, grades (−1 = unjudged),
X and feature names.

Usage
-----
    python -m src.rerank.build --split train --judged --qrels all --name train_judged
    python -m src.rerank.build --split dev --run runs/rrf_ft_pa05_dev.tsv --depth 500 --name dev_rrf_ft
    python -m src.rerank.build --split train+dev --judged --qrels all --name train_dev_judged   # final model
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.baselines.bm25 import INDEX_DIR as BM25_DIR, BM25Index, tokens
from src.data.formula_graph import load_vocabs, opt_graph, slt_graph
from src.data.paths import PROCESSED_DIR, REPO_ROOT, VISUAL_INDEX_DIR
from src.data.topics import load_split_topics
from src.eval.metrics import ranked
from src.eval.runs import read_run
from src.finetune.train import load_train_qrels
from src.model.quick_dev import encode_views, topic_views
from src.model.retrieve import load_checkpoint
from src.pretrain.dataset import edge_type_counts
from src.pretrain.graph_store import REPS, STORE_DIR, GraphStore
from src.rerank.features import FEATURE_NAMES, IdfTable, formula_keys, pair_features, prepare

OUT_DIR = PROCESSED_DIR / "rerank"
EXTRA_FEATURES = ["bm25", "bm25_norm", "gnn_pre", "gnn_pre_norm"]
ALL_FEATURES = FEATURE_NAMES + EXTRA_FEATURES


def load_formulas(visual_ids: set, visual_index_dir: Path = VISUAL_INDEX_DIR) -> Dict[str, Tuple[str, str, str]]:
    """{visual_id: (latex, slt, opt)} for the requested visual ids."""
    from src.data.visual_index import iter_batches

    wanted = pa.array(sorted(visual_ids), type=pa.string())
    out: Dict[str, Tuple[str, str, str]] = {}
    for batch in iter_batches(["visual_id", "latex", "slt", "opt"], batch_size=500_000, out_dir=visual_index_dir):
        hit = batch.filter(pc.is_in(batch.column("visual_id"), value_set=wanted))
        for vid, latex, slt, opt in zip(*(hit.column(c).to_pylist() for c in ("visual_id", "latex", "slt", "opt"))):
            out[vid] = (latex or "", slt, opt)
    return out


def _formula_keys(item: Tuple[str, str]) -> np.ndarray:
    slt, opt = item
    return formula_keys([prepare(slt_graph(slt), trees=False), prepare(opt_graph(opt), trees=False)])


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = (x.min(), x.max()) if len(x) else (0.0, 0.0)
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, help="train/dev/test, or '+'-joined (train+dev) with --judged")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--judged", action="store_true", help="candidates = all judged visual ids")
    source.add_argument("--run", type=Path, help="candidates = top --depth of this run")
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--qrels", default="official", help="official / all / path (grades, and candidates with --judged)")
    parser.add_argument("--pretrained", type=Path, default=REPO_ROOT / "checkpoints/pretrain_rename04/best_rename04.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    parser.add_argument("--visual-index-dir", type=Path, default=VISUAL_INDEX_DIR)
    parser.add_argument("--bm25-dir", type=Path, default=BM25_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--idf-sample", type=int, default=100_000, help="corpus formulas sampled for IDF")
    parser.add_argument("--idf-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="processes for the IDF sample")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    topics = load_split_topics(args.split)
    qrels = load_train_qrels(args.qrels, args.split)

    # Candidate lists, topic by topic
    if args.judged:
        candidates = {t: sorted(qrels[t]) for t in sorted(qrels) if t in topics}
    else:
        run, _ = read_run(args.run)
        candidates = {t: ranked(docs)[:args.depth] for t, docs in sorted(run.items()) if t in topics}
    wanted = {v for vids in candidates.values() for v in vids}
    print(f"{len(candidates)} topics, {sum(map(len, candidates.values())):,} (topic, candidate) pairs", flush=True)

    # Row order shared by the visual index, the graph store and BM25
    store = GraphStore(args.store)
    bm25 = BM25Index.load(args.bm25_dir)
    if len(bm25.doc_ids) != len(store) or bm25.doc_ids[:1000] != store.visual_ids[:1000] \
            or bm25.doc_ids[-1000:] != store.visual_ids[-1000:]:
        raise SystemExit("BM25 index and graph store disagree on row order; rebuild them from the same visual index")
    row = {vid: i for i, vid in enumerate(store.visual_ids) if vid in wanted}
    missing = wanted - set(row)
    if missing:
        print(f"warning: {len(missing)} candidates not in the corpus are skipped", flush=True)
        candidates = {t: [v for v in vids if v in row] for t, vids in candidates.items()}

    # IDF table from a random corpus sample (cached), read in the same pass as the candidates
    idf_path = args.out_dir / f"idf_{args.idf_sample}_{args.idf_seed}.npz"
    sample: List[str] = []
    if not idf_path.exists():
        picks = np.random.default_rng(args.idf_seed).choice(len(store), min(args.idf_sample, len(store)), replace=False)
        sample = [store.visual_ids[i] for i in sorted(picks)]
    formulas = load_formulas(set(row) | set(sample), args.visual_index_dir)
    if idf_path.exists():
        idf = IdfTable.load(idf_path)
        print(f"using IDF table {idf_path} ({idf.n_docs:,} formulas)", flush=True)
    else:
        items = [formulas[v][1:] for v in sample if v in formulas]
        with Pool(args.workers) as pool:
            idf = IdfTable.from_formulas(pool.imap(_formula_keys, items, chunksize=500))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        idf.save(idf_path)
        print(f"IDF table from {idf.n_docs:,} sampled formulas: {len(idf.keys):,} keys → {idf_path} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Candidate formulas → prepared graphs (from the original MathML, so rare symbols match exactly)
    prepared = {v: {"slt": prepare(slt_graph(formulas[v][1]), idf), "opt": prepare(opt_graph(formulas[v][2]), idf)}
                for v in row if v in formulas}
    print(f"prepared {len(prepared):,} candidate formulas ({time.time() - t0:.0f}s)", flush=True)

    # Pretrained-encoder vectors for queries and candidates
    model, ckpt = load_checkpoint(args.pretrained, device)
    vocabs = load_vocabs(Path(ckpt["vocab_path"]))
    n_types = edge_type_counts(vocabs)
    topic_ids = sorted(candidates)
    qviews = topic_views({t: topics[t] for t in topic_ids}, vocabs)
    qvec = dict(zip(topic_ids, encode_views(model, [qviews[t] for t in topic_ids], n_types, device)))
    cand_ids = sorted(row, key=row.get)
    cvec = dict(zip(cand_ids, encode_views(model, [{r: store.get(row[v], r) for r in REPS} for v in cand_ids],
                                           n_types, device)))

    topics_col: List[str] = []
    vids_col: List[str] = []
    grades: List[int] = []
    rows: List[np.ndarray] = []
    for t in topic_ids:
        vids = candidates[t]
        if not vids:
            continue
        topic = topics[t]
        q = {"slt": prepare(slt_graph(topic.slt), idf), "opt": prepare(opt_graph(topic.opt), idf)}
        struct = np.stack([pair_features(q, prepared[v]) for v in vids])
        # BM25 scores of exactly these candidates
        qtf: Dict[str, int] = {}
        for tok in tokens(topic.slt, topic.latex):
            if tok in bm25.vocab:
                qtf[tok] = qtf.get(tok, 0) + 1
        if qtf:
            cols = [bm25.vocab[tok] for tok in qtf]
            sub = bm25.weights[:, cols].tocsr()[[row[v] for v in vids]]
            bm = np.asarray(sub @ np.array(list(qtf.values()), dtype=np.float32)).ravel()
        else:
            bm = np.zeros(len(vids), dtype=np.float32)
        gnn = np.array([float(qvec[t] @ cvec[v]) for v in vids], dtype=np.float32)
        extra = np.stack([bm, _minmax(bm), gnn, _minmax(gnn)], axis=1)
        rows.append(np.concatenate([struct, extra], axis=1))
        topics_col += [t] * len(vids)
        vids_col += vids
        grades += [qrels.get(t, {}).get(v, -1) for v in vids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.npz"
    np.savez_compressed(out, topics=np.array(topics_col), visual_ids=np.array(vids_col), grades=np.array(grades),
                        X=np.concatenate(rows).astype(np.float32), names=np.array(ALL_FEATURES))
    judged = sum(g >= 0 for g in grades)
    print(f"wrote {out}: {len(grades):,} rows × {len(ALL_FEATURES)} features; {judged:,} judged "
          f"({sum(g >= 2 for g in grades):,} relevant) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
