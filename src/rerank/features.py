"""
Structural similarity features between a query formula and a candidate, for reranking.

Both formulas are compared in their SLT and OPT graphs (src.data.formula_graph) at three
levels of strictness, so the reranker can learn how much exact symbols matter:

  exact      node label = tag + symbol                         (x² − y ≠ a² − b)
  unified    variables → VAR, numbers → NUM                    (x² − y = a² − b)
  structure  tag only; operators stay in OPT, whose operator   (x² − y ≈ a³ + b)
             names are tags (plus, times, …)

Per representation and level:
  edge_{recall,precision,f1}   overlap of (parent, edge type, child) tuples, the
                               symbol pairs of Tangent-S; recall = share of the query's
                               tuples found in the candidate
  path_{recall,precision,f1}   the same for two-edge paths (parent, child, grandchild)
Per representation also:
  ted_exact, ted_unified       1 − tree edit distance / larger tree size (graphs read as
                               trees from the root along forward edges, first ≤ 64 nodes)
  size_ratio                   smaller / larger node count
  missing                      1 if either formula lacks this graph (its features are 0)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.formula_graph import FormulaGraph
from src.pretrain.augment import variable_class
from src.rerank.tree_edit import Tree, tree_edit_distance

REPS = ("slt", "opt")
LEVELS = ("exact", "unified", "structure")
MAX_TREE_NODES = 64
_IDENT_TAGS = {"mi", "ci", "csymbol"}
_NUMBER_TAGS = {"mn", "cn"}


def _label(tag: str, symbol: str, level: str) -> str:
    if level == "structure":
        return tag
    if level == "unified":
        if tag in _IDENT_TAGS and variable_class(symbol):
            return f"{tag}:VAR"
        if tag in _NUMBER_TAGS and symbol:
            return f"{tag}:NUM"
    return f"{tag}:{symbol}"


def _feature_names() -> List[str]:
    names = []
    for rep in REPS:
        for level in LEVELS:
            for kind in ("edge", "path"):
                names += [f"{rep}_{level}_{kind}_{m}" for m in ("recall", "precision", "f1")]
        names += [f"{rep}_ted_exact", f"{rep}_ted_unified", f"{rep}_size_ratio", f"{rep}_missing"]
    return names


FEATURE_NAMES = _feature_names()


@dataclass
class Prepared:
    """Per-formula, per-representation precomputation, reused across many comparisons."""
    n_nodes: int
    tuples: Dict[Tuple[str, str], Counter]      # (level, edge|path) → multiset of tuples
    trees: Dict[str, Tree]                      # level → tree for edit distance


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


def prepare(graph: Optional[FormulaGraph]) -> Optional[Prepared]:
    if graph is None or graph.num_nodes == 0:
        return None
    tuples: Dict[Tuple[str, str], Counter] = {}
    trees: Dict[str, Tree] = {}
    out_edges: Dict[int, List[Tuple[int, str]]] = {}
    for a, b, e in graph.edges:
        out_edges.setdefault(a, []).append((b, e))
    for level in LEVELS:
        labels = [_label(t, s, level) for t, s in zip(graph.tags, graph.symbols)]
        tuples[(level, "edge")] = Counter((labels[a], e, labels[b]) for a, b, e in graph.edges)
        tuples[(level, "path")] = Counter((labels[a], e1, labels[b], e2, labels[c])
                                          for a, b, e1 in graph.edges for c, e2 in out_edges.get(b, []))
        if level != "structure":
            trees[level] = _tree(graph, labels)
    return Prepared(graph.num_nodes, tuples, trees)


def _overlap(q: Counter, c: Counter) -> Tuple[float, float, float]:
    nq, nc = sum(q.values()), sum(c.values())
    if nq == 0 and nc == 0:
        return 1.0, 1.0, 1.0  # e.g. two single-node graphs: no tuples, nothing differs
    inter = sum((q & c).values())
    recall = inter / nq if nq else 0.0
    precision = inter / nc if nc else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def _ted_similarity(a: Tree, b: Tree) -> float:
    size = max(len(a[0]), len(b[0]))
    return 1.0 - tree_edit_distance(a, b) / size if size else 1.0


def pair_features(query: Dict[str, Optional[Prepared]], candidate: Dict[str, Optional[Prepared]]) -> np.ndarray:
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
    return np.array(out, dtype=np.float32)
