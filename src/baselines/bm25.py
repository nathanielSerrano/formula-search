"""
BM25 over formula symbols: the sparse baseline, and the first end-to-end test of
the evaluation pipeline.

Each visual id is represented by the symbols of its SLT in reading order (identifier,
number, operator and text leaves, Unicode NFKC-normalised so that 𝑥 and x match),
as unigrams plus adjacent-symbol bigrams. Formulas without SLT fall back to LaTeX
tokens. Scoring is standard Okapi BM25 over a sparse document × term matrix.

Usage (on the server, after `python -m src.data.visual_index`):
    python -m src.baselines.bm25 build
    python -m src.baselines.bm25 search --split dev
    python -m src.eval.evaluate runs/bm25_dev.tsv --split dev
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from src.data.mathml import parse
from src.data.paths import PROCESSED_DIR, REPO_ROOT, VISUAL_INDEX_DIR
from src.data.topics import load_topics
from src.data.visual_index import iter_batches
from src.eval.metrics import MAX_DEPTH, Run
from src.eval.runs import write_run

INDEX_DIR = PROCESSED_DIR / "bm25"
RUNS_DIR = REPO_ROOT / "runs"

_LEAF_TAGS = {"mi", "mn", "mo", "mtext", "ms"}
_INVISIBLE = {"\u2061", "\u2062", "\u2063", "\u2064"}  # function application, invisible times/separator/plus
_LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]|[^\s{}]")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def slt_symbols(slt: Optional[str]) -> Optional[List[str]]:
    """Leaf symbols of an SLT in document order, or None if it cannot be parsed."""
    root = parse(slt)
    if root is None:
        return None
    symbols = []
    for elem in root.iter():
        if elem.tag in _LEAF_TAGS and elem.text:
            s = unicodedata.normalize("NFKC", elem.text).strip()
            if s and s not in _INVISIBLE:
                symbols.append(s)
    return symbols


def latex_symbols(latex: str) -> List[str]:
    return _LATEX_TOKEN_RE.findall(latex or "")


def tokens(slt: Optional[str], latex: str = "") -> List[str]:
    """Unigrams and adjacent bigrams of the formula's symbols."""
    symbols = slt_symbols(slt)
    if symbols is None:
        symbols = latex_symbols(latex)
    return symbols + [f"{a} {b}" for a, b in zip(symbols, symbols[1:])]


def _tokenize_batch(pairs: Sequence) -> List[List[str]]:
    return [tokens(slt, latex) for slt, latex in pairs]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Documents × terms matrix of BM25 term weights (CSC, so query-term columns slice fast)."""

    def __init__(self, weights: sp.csc_matrix, vocab: Dict[str, int], doc_ids: List[str], params: dict):
        self.weights, self.vocab, self.doc_ids, self.params = weights, vocab, doc_ids, params

    @classmethod
    def from_counts(cls, counts: sp.csr_matrix, vocab: Dict[str, int], doc_ids: List[str],
                    k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        n_docs = counts.shape[0]
        doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        avg_len = doc_len.mean() if n_docs else 0.0
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        tf = counts.data.astype(np.float64)
        row_len = np.repeat(doc_len, np.diff(counts.indptr))
        norm = tf + k1 * (1.0 - b + b * row_len / avg_len)
        data = (idf[counts.indices] * tf * (k1 + 1.0) / norm).astype(np.float32)
        weights = sp.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape).tocsc()
        params = {"k1": k1, "b": b, "n_docs": n_docs, "avg_len": float(avg_len), "n_terms": len(vocab)}
        return cls(weights, vocab, doc_ids, params)

    def search(self, query_tokens: List[str], k: int = MAX_DEPTH) -> Dict[str, float]:
        qtf = Counter(t for t in query_tokens if t in self.vocab)
        if not qtf:
            return {}
        cols = [self.vocab[t] for t in qtf]
        scores = self.weights[:, cols] @ np.array([qtf[t] for t in qtf], dtype=np.float32)
        nonzero = np.flatnonzero(scores)
        if len(nonzero) > k:
            nonzero = nonzero[np.argpartition(-scores[nonzero], k - 1)[:k]]
        return {self.doc_ids[i]: float(scores[i]) for i in nonzero}

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(out_dir / "weights.npz", self.weights)
        (out_dir / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False))
        (out_dir / "doc_ids.txt").write_text("\n".join(self.doc_ids))
        (out_dir / "params.json").write_text(json.dumps(self.params, indent=2))

    @classmethod
    def load(cls, out_dir: Path = INDEX_DIR) -> "BM25Index":
        if not (out_dir / "weights.npz").exists():
            raise FileNotFoundError(f"No BM25 index in {out_dir}; run `python -m src.baselines.bm25 build`")
        return cls(sp.load_npz(out_dir / "weights.npz").tocsc(),
                   json.loads((out_dir / "vocab.json").read_text()),
                   (out_dir / "doc_ids.txt").read_text().split("\n"),
                   json.loads((out_dir / "params.json").read_text()))


def build(k1: float, b: float, workers: int, batch_size: int = 20_000,
          visual_index_dir: Path = VISUAL_INDEX_DIR) -> BM25Index:
    """Tokenise the visual index and build the BM25 matrix."""
    vocab: Dict[str, int] = {}
    doc_ids: List[str] = []
    indptr, indices, data = array("q", [0]), array("i"), array("f")
    t0 = time.time()

    def batches():
        for batch in iter_batches(["visual_id", "slt", "latex"], batch_size=batch_size, out_dir=visual_index_dir):
            doc_ids.extend(batch.column("visual_id").to_pylist())
            yield list(zip(batch.column("slt").to_pylist(), batch.column("latex").to_pylist()))

    with Pool(workers) as pool:
        for docs in pool.imap(_tokenize_batch, batches()):
            for toks in docs:
                for term, count in Counter(toks).items():
                    indices.append(vocab.setdefault(term, len(vocab)))
                    data.append(count)
                indptr.append(len(indices))
            print(f"  tokenised {len(indptr) - 1:,} formulas, {len(vocab):,} terms ({time.time() - t0:.0f}s)",
                  flush=True)

    counts = sp.csr_matrix((np.frombuffer(data, dtype=np.float32), np.frombuffer(indices, dtype=np.int32),
                            np.frombuffer(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(vocab)))
    return BM25Index.from_counts(counts, vocab, doc_ids, k1=k1, b=b)


def search_topics(index: BM25Index, split: str, k: int = MAX_DEPTH) -> Run:
    run: Run = {}
    for topic in load_topics(split).values():
        run[topic.topic_id] = index.search(tokens(topic.slt, topic.latex), k)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the BM25 index from the visual index")
    p_build.add_argument("--k1", type=float, default=1.2)
    p_build.add_argument("--b", type=float, default=0.75)
    p_build.add_argument("--workers", type=int, default=8)
    p_build.add_argument("--out-dir", type=Path, default=INDEX_DIR)

    p_search = sub.add_parser("search", help="retrieve for every topic of a split and write a run file")
    p_search.add_argument("--split", required=True, help="train/dev/test or arqmath1/2/3")
    p_search.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    p_search.add_argument("--out", type=Path, help="run file (default runs/bm25_<split>.tsv)")
    p_search.add_argument("--run-id", default="bm25")
    args = parser.parse_args()

    if args.command == "build":
        index = build(args.k1, args.b, args.workers)
        index.save(args.out_dir)
        print(f"BM25 index saved to {args.out_dir}: {index.params}")
    else:
        index = BM25Index.load(args.index_dir)
        run = search_topics(index, args.split)
        out = args.out or RUNS_DIR / f"{args.run_id}_{args.split}.tsv"
        write_run(run, out, args.run_id)
        empty = [t for t, docs in run.items() if not docs]
        print(f"Wrote {out} ({len(run)} topics" + (f", {len(empty)} with no results: {empty[:5]}" if empty else "") + ")")


if __name__ == "__main__":
    main()
