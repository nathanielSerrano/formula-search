"""
Precomputed SLT/OPT graphs for every visual id, as flat integer arrays on disk.

Contrastive pretraining reads millions of formulas per epoch; parsing MathML each time
would dominate training. This module converts the visual index once (with the
vocabularies from `python -m src.data.graph_stats`) into memory-mapped arrays:

    data/processed/pretrain_graphs/
      meta.json                    counts, dtypes, max_nodes, vocab path
      visual_ids.txt               row order
      {slt,opt}_node_tag.bin       int16   tag id per node
      {slt,opt}_node_symbol.bin    int32   symbol id per node
      {slt,opt}_node_offsets.bin   int64   formula i owns nodes [off[i], off[i+1])
      {slt,opt}_edge_src.bin       int16   local node index (forward edges only)
      {slt,opt}_edge_dst.bin       int16
      {slt,opt}_edge_type.bin      int8    forward edge type id
      {slt,opt}_edge_offsets.bin   int64

A formula with no graph for a representation owns zero nodes there. Graphs are
truncated to max_nodes (default 256; cuts ~0.03% of SLT and ~0.04% of OPT graphs).

Build (on the server, after graph_stats):
    python -m src.pretrain.graph_store
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data.formula_graph import FormulaGraph, GraphVocab, load_vocabs, opt_graph, slt_graph
from src.data.paths import PROCESSED_DIR, VISUAL_INDEX_DIR
from src.data.visual_index import iter_batches

STORE_DIR = PROCESSED_DIR / "pretrain_graphs"
VOCAB_PATH = PROCESSED_DIR / "graph_vocab.json"
REPS = ("slt", "opt")
_BUILDERS = {"slt": slt_graph, "opt": opt_graph}
_FIELDS = {
    "node_tag": np.int16, "node_symbol": np.int32,
    "edge_src": np.int16, "edge_dst": np.int16, "edge_type": np.int8,
}

# Worker state (set once per process by _init_worker).
_VOCABS: Dict[str, GraphVocab] = {}
_MAX_NODES = 256


def _init_worker(vocab_path: str, max_nodes: int) -> None:
    global _VOCABS, _MAX_NODES
    _VOCABS = load_vocabs(Path(vocab_path))
    _MAX_NODES = max_nodes


def _encode_batch(pairs: Sequence) -> Dict[str, Dict[str, np.ndarray]]:
    """Concatenated arrays for one batch of (slt, opt) strings, per representation."""
    out = {}
    for r, rep in enumerate(REPS):
        vocab = _VOCABS[rep]
        parts: Dict[str, List[np.ndarray]] = {k: [] for k in _FIELDS}
        n_nodes, n_edges, truncated = [], [], 0
        for pair in pairs:
            graph = _BUILDERS[rep](pair[r]) if pair[r] else None
            if graph is None:
                n_nodes.append(0)
                n_edges.append(0)
                continue
            arrays = vocab.encode(graph, _MAX_NODES)
            truncated += int(arrays["truncated"])
            forward = len(arrays["edge_type"]) // 2  # encode() lists forward edges first
            parts["node_tag"].append(arrays["tag"])
            parts["node_symbol"].append(arrays["symbol"])
            parts["edge_src"].append(arrays["edge_index"][0, :forward])
            parts["edge_dst"].append(arrays["edge_index"][1, :forward])
            parts["edge_type"].append(arrays["edge_type"][:forward])
            n_nodes.append(len(arrays["tag"]))
            n_edges.append(forward)
        rep_out = {k: (np.concatenate(v) if v else np.zeros(0)).astype(_FIELDS[k]) for k, v in parts.items()}
        rep_out["n_nodes"] = np.array(n_nodes, dtype=np.int64)
        rep_out["n_edges"] = np.array(n_edges, dtype=np.int64)
        rep_out["truncated"] = np.array(truncated)
        out[rep] = rep_out
    return out


def build(out_dir: Path = STORE_DIR, vocab_path: Path = VOCAB_PATH, visual_index_dir: Path = VISUAL_INDEX_DIR,
          max_nodes: int = 256, workers: int = 8, batch_size: int = 20_000) -> dict:
    if max_nodes > np.iinfo(np.int16).max:
        raise ValueError("max_nodes must fit in int16")
    load_vocabs(vocab_path)  # fail early if missing
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {(rep, k): open(out_dir / f"{rep}_{k}.bin", "wb") for rep in REPS for k in _FIELDS}
    node_offsets = {rep: [np.zeros(1, dtype=np.int64)] for rep in REPS}
    edge_offsets = {rep: [np.zeros(1, dtype=np.int64)] for rep in REPS}
    totals = {rep: {"nodes": 0, "edges": 0, "graphs": 0, "truncated": 0} for rep in REPS}
    visual_ids: List[str] = []
    t0 = time.time()

    def batches():
        for batch in iter_batches(["visual_id", "slt", "opt"], batch_size=batch_size, out_dir=visual_index_dir):
            visual_ids.extend(batch.column("visual_id").to_pylist())
            yield list(zip(batch.column("slt").to_pylist(), batch.column("opt").to_pylist()))

    try:
        with Pool(workers, initializer=_init_worker, initargs=(str(vocab_path), max_nodes)) as pool:
            for result in pool.imap(_encode_batch, batches()):  # ordered, so rows match visual_ids
                for rep in REPS:
                    r = result[rep]
                    for k in _FIELDS:
                        files[(rep, k)].write(r[k].tobytes())
                    t = totals[rep]
                    node_offsets[rep].append(t["nodes"] + np.cumsum(r["n_nodes"]))
                    edge_offsets[rep].append(t["edges"] + np.cumsum(r["n_edges"]))
                    t["nodes"] += int(r["n_nodes"].sum())
                    t["edges"] += int(r["n_edges"].sum())
                    t["graphs"] += int((r["n_nodes"] > 0).sum())
                    t["truncated"] += int(r["truncated"])
                n = sum(len(o) for o in node_offsets["slt"][1:])
                print(f"  {n:,} formulas ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for f in files.values():
            f.close()

    for rep in REPS:
        np.concatenate(node_offsets[rep]).tofile(out_dir / f"{rep}_node_offsets.bin")
        np.concatenate(edge_offsets[rep]).tofile(out_dir / f"{rep}_edge_offsets.bin")
    (out_dir / "visual_ids.txt").write_text("\n".join(visual_ids))
    meta = {"n_formulas": len(visual_ids), "max_nodes": max_nodes, "vocab": str(vocab_path),
            "dtypes": {k: np.dtype(v).name for k, v in _FIELDS.items()}, "totals": totals,
            "seconds": round(time.time() - t0)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def arrays_from_graph(graph: Optional[FormulaGraph], vocab: GraphVocab,
                      max_nodes: int = 256) -> Optional[Dict[str, np.ndarray]]:
    """The GraphStore.get format (forward edges only) for a graph built on the fly, e.g. a query."""
    if graph is None:
        return None
    arrays = vocab.encode(graph, max_nodes)
    forward = len(arrays["edge_type"]) // 2
    return {"tag": arrays["tag"], "symbol": arrays["symbol"], "src": arrays["edge_index"][0, :forward],
            "dst": arrays["edge_index"][1, :forward], "type": arrays["edge_type"][:forward]}


class GraphStore:
    """Read access to a built store. `get(i, rep)` returns a formula's arrays (None if it has no graph)."""

    def __init__(self, out_dir: Path = STORE_DIR):
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No graph store in {out_dir}; run `python -m src.pretrain.graph_store`")
        self.meta = json.loads(meta_path.read_text())
        self.dir = out_dir
        self.n = self.meta["n_formulas"]
        self._arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for rep in REPS:
            arr = {k: np.memmap(out_dir / f"{rep}_{k}.bin", dtype=v, mode="r") for k, v in _FIELDS.items()}
            for k in ("node_offsets", "edge_offsets"):
                arr[k] = np.memmap(out_dir / f"{rep}_{k}.bin", dtype=np.int64, mode="r")
            self._arrays[rep] = arr
        self._visual_ids: Optional[List[str]] = None

    def __len__(self) -> int:
        return self.n

    @property
    def visual_ids(self) -> List[str]:
        if self._visual_ids is None:
            self._visual_ids = (self.dir / "visual_ids.txt").read_text().split("\n")
        return self._visual_ids

    def num_nodes(self, rep: str) -> np.ndarray:
        return np.diff(self._arrays[rep]["node_offsets"])

    def get(self, i: int, rep: str) -> Optional[Dict[str, np.ndarray]]:
        a = self._arrays[rep]
        n0, n1 = a["node_offsets"][i], a["node_offsets"][i + 1]
        if n0 == n1:
            return None
        e0, e1 = a["edge_offsets"][i], a["edge_offsets"][i + 1]
        return {
            "tag": np.asarray(a["node_tag"][n0:n1], dtype=np.int64),
            "symbol": np.asarray(a["node_symbol"][n0:n1], dtype=np.int64),
            "src": np.asarray(a["edge_src"][e0:e1], dtype=np.int64),
            "dst": np.asarray(a["edge_dst"][e0:e1], dtype=np.int64),
            "type": np.asarray(a["edge_type"][e0:e1], dtype=np.int64),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=STORE_DIR)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--visual-index-dir", type=Path, default=VISUAL_INDEX_DIR)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    meta = build(args.out_dir, args.vocab, args.visual_index_dir, args.max_nodes, args.workers)
    print(f"\nGraph store written to {args.out_dir} in {meta['seconds']}s: {meta['n_formulas']:,} formulas")
    for rep, t in meta["totals"].items():
        print(f"  {rep.upper()}: {t['graphs']:,} graphs, {t['nodes']:,} nodes, {t['edges']:,} edges, "
              f"{t['truncated']:,} truncated to {args.max_nodes} nodes")


if __name__ == "__main__":
    main()
