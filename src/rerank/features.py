"""
Structural similarity features between a query formula and a candidate, for reranking.

Both formulas are compared in their SLT and OPT graphs (src.data.formula_graph) at several
levels of strictness, so the reranker can learn how much exact symbols matter:

  exact      node label = tag + symbol                         (x² − y ≠ a² − b)
  alpha      variables renamed V1, V2, … in order of first     (x·x = a·a ≠ x·y = a·b)
             appearance in the canonical tree; numbers exact
  unified    variables → VAR, numbers → NUM                    (x² − y = a² − b)
  structure  tag only; operators stay in OPT, whose operator   (x² − y ≈ a³ + b)
             names are tags (plus, times, …)

Base group (BASE_FEATURE_NAMES), per representation and level exact / unified / structure:
  edge_{recall,precision,f1}   overlap of (parent, edge type, child) tuples, the
                               symbol pairs of Tangent-S; recall = share of the query's
                               tuples found in the candidate
  path_{recall,precision,f1}   the same for two-edge paths (parent, child, grandchild)
Per representation also:
  ted_exact, ted_unified       1 − tree edit distance / larger tree size (graphs read as
                               trees from the root along forward edges, first ≤ 64 nodes)
  size_ratio                   smaller / larger node count
  missing                      1 if either formula lacks this graph (its features are 0)

Proto group (PROTO_FEATURE_NAMES): the ideas of the earlier prototype's hand-written
reranker (src/task3/eval/phase4_structural_reranker.py, removed; see commit b0eb6c2), as learnable features:
  alpha_{edge,path}_*          the overlaps above at the alpha level
  <level>_path3_*              three-edge paths, at all four levels
  idf_<level>_<kind>_{recall,precision}
                               edge / path overlap weighted by corpus IDF (IdfTable), so a
                               rare matching tuple counts more than a common one; exact,
                               alpha and unified levels
  ted_{exact,alpha}_canon      tree edit similarity on the canonical tree, where operands
                               of commutative operators ("arg" edges) are sorted by shape,
                               so a + b and b + a are identical
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data.formula_graph import FormulaGraph
from src.pretrain.augment import variable_class
from src.rerank.tree_edit import Tree, tree_edit_distance

REPS = ("slt", "opt")
LEVELS = ("exact", "unified", "structure")
PROTO_LEVELS = ("exact", "alpha", "unified", "structure")
IDF_LEVELS = ("exact", "alpha", "unified")
MAX_TREE_NODES = 64
_IDENT_TAGS = {"mi", "ci", "csymbol"}
_NUMBER_TAGS = {"mn", "cn"}
_RPF = ("recall", "precision", "f1")


def _label(tag: str, symbol: str, level: str) -> str:
    if level == "structure":
        return tag
    if level == "unified":
        if tag in _IDENT_TAGS and variable_class(symbol):
            return f"{tag}:VAR"
        if tag in _NUMBER_TAGS and symbol:
            return f"{tag}:NUM"
    return f"{tag}:{symbol}"


def _base_names() -> List[str]:
    names = []
    for rep in REPS:
        for level in LEVELS:
            for kind in ("edge", "path"):
                names += [f"{rep}_{level}_{kind}_{m}" for m in _RPF]
        names += [f"{rep}_ted_exact", f"{rep}_ted_unified", f"{rep}_size_ratio", f"{rep}_missing"]
    return names


def _proto_names() -> List[str]:
    names = []
    for rep in REPS:
        names += [f"{rep}_alpha_{kind}_{m}" for kind in ("edge", "path") for m in _RPF]
        names += [f"{rep}_{level}_path3_{m}" for level in PROTO_LEVELS for m in _RPF]
        names += [f"{rep}_idf_{level}_{kind}_{m}" for level in IDF_LEVELS for kind in ("edge", "path")
                  for m in ("recall", "precision")]
        names += [f"{rep}_ted_exact_canon", f"{rep}_ted_alpha_canon"]
    return names


BASE_FEATURE_NAMES = _base_names()
PROTO_FEATURE_NAMES = _proto_names()
FEATURE_NAMES = BASE_FEATURE_NAMES + PROTO_FEATURE_NAMES
_PROTO_PER_REP = len(PROTO_FEATURE_NAMES) // len(REPS)


class IdfTable:
    """Document frequencies of hashed (level, kind, tuple) keys over a sample of formulas."""

    def __init__(self, keys: np.ndarray, df: np.ndarray, n_docs: int):
        self.keys, self.df, self.n_docs = keys, df, n_docs

    @staticmethod
    def key(level: str, kind: str, tup: Tuple[str, ...]) -> int:
        digest = hashlib.blake2b("\x1f".join((level, kind) + tup).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    @classmethod
    def from_formulas(cls, key_sets: Iterable[np.ndarray]) -> "IdfTable":
        """key_sets: one array of distinct keys per formula (formula_keys)."""
        sets = list(key_sets)
        keys, df = np.unique(np.concatenate(sets) if sets else np.zeros(0, np.uint64), return_counts=True)
        return cls(keys, df, len(sets))

    def idf(self, keys: np.ndarray) -> np.ndarray:
        """BM25-style IDF, log(1 + (N − df + 0.5) / (df + 0.5)); unseen keys get the maximum."""
        idx = np.minimum(np.searchsorted(self.keys, keys), max(len(self.keys) - 1, 0))
        found = (self.keys[idx] == keys) if len(self.keys) else np.zeros(len(keys), bool)
        df = np.where(found, self.df[idx] if len(self.keys) else 0, 0).astype(np.float64)
        return np.log1p((self.n_docs - df + 0.5) / (df + 0.5))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, keys=self.keys, df=self.df, n_docs=self.n_docs)

    @classmethod
    def load(cls, path: Path) -> "IdfTable":
        d = np.load(path)
        return cls(d["keys"], d["df"], int(d["n_docs"]))


@dataclass
class Prepared:
    """Per-formula, per-representation precomputation, reused across many comparisons."""
    n_nodes: int
    tuples: Dict[Tuple[str, str], Counter]      # (level, edge|path|path3) → multiset of tuples
    trees: Dict[str, Tree]                      # level → tree for edit distance ("<level>_canon": canonical)
    weights: Dict[Tuple[str, str], Dict[tuple, float]] = field(default_factory=dict)  # IDF per tuple


def _tree(graph: FormulaGraph, labels: Sequence[str], max_nodes: int = MAX_TREE_NODES) -> Tree:
    children: Dict[int, List[int]] = {}
    for a, b, _ in graph.edges:
        children.setdefault(a, []).append(b)
    order, index, kids, seen = [], {}, [], set()
    stack = [0]
    while stack and len(order) < max_nodes:  # depth-first, each node once (first parent wins)
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        index[node] = len(order)
        order.append(node)
        stack.extend(reversed(children.get(node, [])))
    tree_children: List[List[int]] = [[] for _ in order]
    placed = set()
    for node in order:
        for c in children.get(node, []):
            if c in index and c not in placed and index[c] > index[node]:
                tree_children[index[node]].append(index[c])
                placed.add(c)
    return [labels[n] for n in order], tree_children


def _canonical(graph: FormulaGraph, exact: Sequence[str], unified: Sequence[str]) -> Tuple[List[int], Dict[int, List[int]]]:
    """Nodes reachable from the root in canonical preorder, and each node's children in canonical
    order: children linked by an unordered "arg" edge are sorted by subtree shape (unified labels,
    then exact labels); positional children keep their order."""
    out: Dict[int, List[Tuple[int, str]]] = {}
    for a, b, e in graph.edges:
        out.setdefault(a, []).append((b, e))
    found, linked, seen, stack = [], {}, {0}, [0]
    while stack:  # spanning tree from the root, first link wins
        node = stack.pop()
        found.append(node)
        linked[node] = [(c, e) for c, e in out.get(node, []) if c not in seen]
        seen.update(c for c, _ in linked[node])
        stack.extend(c for c, _ in linked[node])
    kids: Dict[int, List[int]] = {}
    shape: Dict[int, Tuple[str, str]] = {}
    for node in reversed(found):  # every node is found after its parent → children first
        unordered = iter(sorted((c for c, e in linked[node] if e == "arg"), key=shape.__getitem__))
        kids[node] = [next(unordered) if e == "arg" else c for c, e in linked[node]]
        shape[node] = (unified[node] + "(" + ",".join(shape[c][0] for c in kids[node]) + ")",
                       exact[node] + "(" + ",".join(shape[c][1] for c in kids[node]) + ")")
    preorder, stack = [], [0]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(reversed(kids[node]))
    return preorder, kids


def _alpha_labels(graph: FormulaGraph, exact: Sequence[str], preorder: Sequence[int]) -> List[str]:
    """Variables renamed V1, V2, … by first appearance in canonical preorder (unreached nodes last)."""
    names: Dict[str, int] = {}
    reached = set(preorder)
    for n in list(preorder) + [n for n in range(graph.num_nodes) if n not in reached]:
        if graph.tags[n] in _IDENT_TAGS and variable_class(graph.symbols[n]):
            names.setdefault(graph.symbols[n], len(names) + 1)
    return [f"{t}:V{names[s]}" if t in _IDENT_TAGS and s in names else exact[n]
            for n, (t, s) in enumerate(zip(graph.tags, graph.symbols))]


def _canonical_tree(preorder: Sequence[int], kids: Dict[int, List[int]], labels: Sequence[str],
                    max_nodes: int = MAX_TREE_NODES) -> Tree:
    nodes = preorder[:max_nodes]  # a preorder prefix contains every node's parent
    index = {n: i for i, n in enumerate(nodes)}
    return [labels[n] for n in nodes], [[index[c] for c in kids[n] if c in index] for n in nodes]


def prepare(graph: Optional[FormulaGraph], idf: Optional[IdfTable] = None, trees: bool = True) -> Optional[Prepared]:
    """Tuples (and trees) of one graph; with `idf`, IDF weights for the proto group's weighted
    overlaps (without, every tuple weighs 1). trees=False skips the trees (IDF sampling)."""
    if graph is None or graph.num_nodes == 0:
        return None
    tuples: Dict[Tuple[str, str], Counter] = {}
    tree_map: Dict[str, Tree] = {}
    out_edges: Dict[int, List[Tuple[int, str]]] = {}
    for a, b, e in graph.edges:
        out_edges.setdefault(a, []).append((b, e))
    labels = {level: [_label(t, s, level) for t, s in zip(graph.tags, graph.symbols)] for level in LEVELS}
    preorder, kids = _canonical(graph, labels["exact"], labels["unified"])
    labels["alpha"] = _alpha_labels(graph, labels["exact"], preorder)
    for level in PROTO_LEVELS:
        lab = labels[level]
        tuples[(level, "edge")] = Counter((lab[a], e, lab[b]) for a, b, e in graph.edges)
        tuples[(level, "path")] = Counter((lab[a], e1, lab[b], e2, lab[c])
                                          for a, b, e1 in graph.edges for c, e2 in out_edges.get(b, []))
        tuples[(level, "path3")] = Counter((lab[a], e1, lab[b], e2, lab[c], e3, lab[d])
                                           for a, b, e1 in graph.edges for c, e2 in out_edges.get(b, [])
                                           for d, e3 in out_edges.get(c, []))
        if trees and level in ("exact", "unified"):
            tree_map[level] = _tree(graph, lab)
        if trees and level in ("exact", "alpha"):
            tree_map[f"{level}_canon"] = _canonical_tree(preorder, kids, lab)
    weights: Dict[Tuple[str, str], Dict[tuple, float]] = {}
    for level in IDF_LEVELS:
        for kind in ("edge", "path"):
            distinct = list(tuples[(level, kind)])
            if idf is None:
                w = np.ones(len(distinct))
            else:
                w = idf.idf(np.array([IdfTable.key(level, kind, t) for t in distinct], dtype=np.uint64))
            weights[(level, kind)] = dict(zip(distinct, w.tolist()))
    return Prepared(graph.num_nodes, tuples, tree_map, weights)


def formula_keys(prepared: Sequence[Optional[Prepared]]) -> np.ndarray:
    """Distinct IDF keys of one formula (all its representations), for IdfTable.from_formulas."""
    keys = {IdfTable.key(level, kind, t) for p in prepared if p is not None
            for level in IDF_LEVELS for kind in ("edge", "path") for t in p.tuples[(level, kind)]}
    return np.array(sorted(keys), dtype=np.uint64)


def _overlap(q: Counter, c: Counter) -> Tuple[float, float, float]:
    nq, nc = sum(q.values()), sum(c.values())
    if nq == 0 and nc == 0:
        return 1.0, 1.0, 1.0  # e.g. two single-node graphs: no tuples, nothing differs
    inter = sum((q & c).values())
    recall = inter / nq if nq else 0.0
    precision = inter / nc if nc else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def _idf_overlap(q: Counter, qw: Dict[tuple, float], c: Counter, cw: Dict[tuple, float]) -> Tuple[float, float]:
    nq = sum(n * qw[t] for t, n in q.items())
    nc = sum(n * cw[t] for t, n in c.items())
    if nq == 0 and nc == 0:
        return 1.0, 1.0
    inter = sum(min(n, c[t]) * qw[t] for t, n in q.items() if t in c)
    return (inter / nq if nq else 0.0), (inter / nc if nc else 0.0)


def _ted_similarity(a: Tree, b: Tree) -> float:
    size = max(len(a[0]), len(b[0]))
    return 1.0 - tree_edit_distance(a, b) / size if size else 1.0


def _base_features(query: Dict[str, Optional[Prepared]], candidate: Dict[str, Optional[Prepared]]) -> List[float]:
    out: List[float] = []
    for rep in REPS:
        q, c = query.get(rep), candidate.get(rep)
        if q is None or c is None:
            out += [0.0] * (len(LEVELS) * 2 * 3 + 3) + [1.0]
            continue
        for level in LEVELS:
            for kind in ("edge", "path"):
                out += _overlap(q.tuples[(level, kind)], c.tuples[(level, kind)])
        out += [_ted_similarity(q.trees["exact"], c.trees["exact"]),
                _ted_similarity(q.trees["unified"], c.trees["unified"]),
                min(q.n_nodes, c.n_nodes) / max(q.n_nodes, c.n_nodes), 0.0]
    return out


def _proto_features(query: Dict[str, Optional[Prepared]], candidate: Dict[str, Optional[Prepared]]) -> List[float]:
    out: List[float] = []
    for rep in REPS:
        q, c = query.get(rep), candidate.get(rep)
        if q is None or c is None:  # the base group's missing flag covers this
            out += [0.0] * _PROTO_PER_REP
            continue
        for kind in ("edge", "path"):
            out += _overlap(q.tuples[("alpha", kind)], c.tuples[("alpha", kind)])
        for level in PROTO_LEVELS:
            out += _overlap(q.tuples[(level, "path3")], c.tuples[(level, "path3")])
        for level in IDF_LEVELS:
            for kind in ("edge", "path"):
                key = (level, kind)
                out += _idf_overlap(q.tuples[key], q.weights[key], c.tuples[key], c.weights[key])
        out += [_ted_similarity(q.trees["exact_canon"], c.trees["exact_canon"]),
                _ted_similarity(q.trees["alpha_canon"], c.trees["alpha_canon"])]
    return out


def pair_features(query: Dict[str, Optional[Prepared]], candidate: Dict[str, Optional[Prepared]]) -> np.ndarray:
    """FEATURE_NAMES values: the base group, then the proto group."""
    return np.array(_base_features(query, candidate) + _proto_features(query, candidate), dtype=np.float32)
