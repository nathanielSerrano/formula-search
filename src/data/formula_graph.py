"""
Formula graphs: SLT and OPT trees with symbol-labelled nodes and typed edges.

A formula is converted from canonical MathML (src.data.mathml) into a FormulaGraph:
every node carries a tag and a symbol, every edge a relation type.

SLT (Presentation MathML → symbol layout tree, in the style of Tangent's SLTs)
  Nodes are the visible symbols. Layout wrappers (mrow, mstyle, mpadded, …) and
  invisible operators (function application, invisible times) are dropped, so x²−y
  becomes  x —next→ − —next→ y  with  x —sup→ 2.  Structures without a symbol of
  their own get a structural node: mfrac (above/below), msqrt and mroot (within,
  index), mtable/mtr (row, cell). Scripts attach to the last baseline symbol of
  their base, as they do on the page.

OPT (Content MathML → operator tree)
  Each apply becomes a node named by its operator (plus, divide, csymbol
  "superscript", …) with edges to its arguments. Arguments of commutative
  operators use one unordered "arg" type; all others are positional (arg1, arg2, …),
  so a−b and b−a differ. An operator that is itself an expression (e.g. a
  subscripted ∑) hangs off an "apply" node by an "op" edge. Containers (list, set,
  vector, matrix, interval, …) have positional children; interval's symbol is its
  closure, so [a,b] and (a,b) differ.

Symbols are Unicode NFKC-normalised, so the italic 𝑥 of OPT and the x of SLT
match. GraphVocab maps tags, symbols and edge types to ids (rare symbols → <unk>,
unseen numbers → length buckets; ids 0/1 are <pad>/<unk>, never shared) and adds
a reverse type for every edge so messages can flow both ways without losing
direction. `to_pyg` builds the PyTorch Geometric input.
"""

from __future__ import annotations

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.data.mathml import parse

MAX_SYMBOL_LEN = 30

# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------


@dataclass
class FormulaGraph:
    tags: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)  # (parent, child, type)

    def add(self, tag: str, symbol: str = "") -> int:
        self.tags.append(tag)
        self.symbols.append(symbol)
        return len(self.tags) - 1

    def link(self, src: Optional[int], dst: Optional[int], etype: str) -> None:
        if src is not None and dst is not None:
            self.edges.append((src, dst, etype))

    @property
    def num_nodes(self) -> int:
        return len(self.tags)

    def describe(self) -> List[str]:
        """Readable edge list, e.g. ['x -sup-> 2', 'x -next-> −']; for debugging and tests."""
        name = [f"{s or t}" for t, s in zip(self.tags, self.symbols)]
        return [f"{name[a]} -{e}-> {name[b]}" for a, b, e in self.edges] or name


def normalize_symbol(text: Optional[str]) -> str:
    s = " ".join(unicodedata.normalize("NFKC", text or "").split())
    return s[:MAX_SYMBOL_LEN]


# ---------------------------------------------------------------------------
# SLT
# ---------------------------------------------------------------------------

SLT_EDGE_TYPES = ("next", "sup", "sub", "presup", "presub", "over", "under",
                  "above", "below", "within", "index", "row", "cell")

_SLT_LEAVES = {"mi", "mn", "mo", "mtext", "ms"}
_SLT_SKIP = {"mspace", "none", "maligngroup", "malignmark", "mphantom", "mprescripts"}
_INVISIBLE = {"⁡", "⁢", "⁣", "⁤", ""}

Span = Optional[Tuple[int, int]]  # (first, last) baseline node of a subexpression


class _SLTBuilder:
    def __init__(self):
        self.g = FormulaGraph()

    def row(self, elems: Iterable[ET.Element]) -> Span:
        """Convert elements laid out left to right, chaining them with next edges."""
        first = last = None
        for elem in elems:
            span = self.span(elem)
            if span is None:
                continue
            if last is None:
                first = span[0]
            else:
                self.g.link(last, span[0], "next")
            last = span[1]
        return None if first is None else (first, last)

    def _base(self, elem: ET.Element, tag: str) -> Tuple[int, int]:
        span = self.span(elem)
        if span is None:  # empty base, e.g. {}^2
            node = self.g.add(tag)
            return node, node
        return span

    def _structure(self, tag: str, symbol: str, parts: List[Tuple[ET.Element, str]]) -> Span:
        node = self.g.add(tag, symbol)
        for elem, etype in parts:
            span = self.span(elem)
            if span is not None:
                self.g.link(node, span[0], etype)
        return node, node

    def span(self, elem: ET.Element) -> Span:
        tag, kids = elem.tag, list(elem)

        if tag in _SLT_LEAVES and not kids:
            symbol = normalize_symbol(elem.text)
            return None if symbol in _INVISIBLE else (self.g.add(tag, symbol),) * 2
        if tag in _SLT_SKIP:
            return None

        if tag in ("msup", "msub", "msubsup", "munder", "mover", "munderover") and len(kids) >= 2:
            base = self._base(kids[0], tag)
            script_types = {"msup": ["sup"], "msub": ["sub"], "msubsup": ["sub", "sup"],
                            "munder": ["under"], "mover": ["over"], "munderover": ["under", "over"]}[tag]
            for script, etype in zip(kids[1:], script_types):
                span = self.span(script)
                if span is not None:
                    self.g.link(base[1], span[0], etype)
            return base

        if tag == "mmultiscripts" and kids:
            base = self._base(kids[0], tag)
            pre = False
            scripts = kids[1:]
            i = 0
            while i < len(scripts):
                if scripts[i].tag == "mprescripts":
                    pre, i = True, i + 1
                    continue
                for script, etype in zip(scripts[i:i + 2], ("presub", "presup") if pre else ("sub", "sup")):
                    span = self.span(script)
                    if span is not None:
                        self.g.link(base[1], span[0], etype)
                i += 2
            return base

        if tag == "mfrac" and len(kids) == 2:
            return self._structure("mfrac", "", [(kids[0], "above"), (kids[1], "below")])
        if tag == "msqrt":
            node = self.g.add("msqrt", "√")
            self.g.link(node, (self.row(kids) or (None,))[0], "within")
            return node, node
        if tag == "mroot" and len(kids) == 2:
            return self._structure("mroot", "√", [(kids[0], "within"), (kids[1], "index")])

        if tag == "mtable":
            table = self.g.add("mtable")
            prev_row = None
            for tr in kids:
                cells = [c for c in tr if c.tag == "mtd"] if tr.tag in ("mtr", "mlabeledtr") else [tr]
                row = self.g.add("mtr")
                self.g.link(table, row, "row")
                self.g.link(prev_row, row, "next")
                prev_row, prev_cell = row, None
                for cell in cells:
                    span = self.row(list(cell)) if cell.tag == "mtd" else self.span(cell)
                    if span is None:
                        continue
                    self.g.link(row, span[0], "cell")
                    self.g.link(prev_cell, span[0], "next")
                    prev_cell = span[1]
            return table, table

        if tag == "mfenced":
            opener = self.g.add("mo", normalize_symbol(elem.get("open", "(")))
            separator = normalize_symbol(elem.get("separators", ",")[:1])
            last = opener
            for k, kid in enumerate(kids):
                if k and separator:
                    sep = self.g.add("mo", separator)
                    self.g.link(last, sep, "next")
                    last = sep
                span = self.span(kid)
                if span is not None:
                    self.g.link(last, span[0], "next")
                    last = span[1]
            closer = self.g.add("mo", normalize_symbol(elem.get("close", ")")))
            self.g.link(last, closer, "next")
            return opener, closer

        if not kids and elem.text and elem.text.strip():  # unknown leaf with content
            return (self.g.add(tag, normalize_symbol(elem.text)),) * 2
        # math, mrow, mstyle, mpadded, merror, menclose, maction, mtd, …: lay children out in a row
        return self.row(kids)


def slt_graph(xml: Optional[str]) -> Optional[FormulaGraph]:
    """SLT graph of a Presentation MathML string, or None if it is unparseable or empty."""
    root = parse(xml)
    if root is None:
        return None
    builder = _SLTBuilder()
    builder.row([root])
    return builder.g if builder.g.num_nodes else None


# ---------------------------------------------------------------------------
# OPT
# ---------------------------------------------------------------------------

MAX_POSITIONAL_ARGS = 4
OPT_EDGE_TYPES = ("op", "arg", *(f"arg{i}" for i in range(1, MAX_POSITIONAL_ARGS + 1)), "argN",
                  "bvar", "lowlimit", "uplimit", "degree", "condition", "domainofapplication", "logbase", "momentabout")

_COMMUTATIVE = {"plus", "times", "eq", "neq", "approx", "equivalent", "and", "or", "xor",
                "union", "intersect", "gcd", "lcm", "max", "min", "set"}
_QUALIFIERS = {"bvar", "lowlimit", "uplimit", "degree", "condition", "domainofapplication", "logbase", "momentabout"}
_SYMBOL_ATTRIBUTE = {"interval": "closure"}


def _positional(i: int) -> str:
    return f"arg{i}" if i <= MAX_POSITIONAL_ARGS else "argN"


class _OPTBuilder:
    def __init__(self):
        self.g = FormulaGraph()

    def _symbol(self, elem: ET.Element) -> str:
        if elem.tag in _SYMBOL_ATTRIBUTE:
            return normalize_symbol(elem.get(_SYMBOL_ATTRIBUTE[elem.tag], ""))
        return normalize_symbol(elem.text)

    def _children(self, node: int, kids: List[ET.Element], unordered: bool) -> None:
        position = 0
        for kid in kids:
            if kid.tag in _QUALIFIERS:
                for content in kid:
                    self.g.link(node, self.node(content), kid.tag)
                continue
            position += 1
            self.g.link(node, self.node(kid), "arg" if unordered else _positional(position))

    def node(self, elem: ET.Element) -> Optional[int]:
        tag, kids = elem.tag, list(elem)

        if tag == "apply":
            if not kids:
                return None
            head, args = kids[0], kids[1:]
            if len(head) == 0:
                node = self.g.add(head.tag, self._symbol(head))
            else:  # the operator is itself an expression
                node = self.g.add("apply")
                self.g.link(node, self.node(head), "op")
            self._children(node, args, unordered=head.tag in _COMMUTATIVE)
            return node

        if tag == "math" and len(kids) == 1:
            return self.node(kids[0])

        node = self.g.add(tag, self._symbol(elem))
        self._children(node, kids, unordered=tag in _COMMUTATIVE)
        return node


def opt_graph(xml: Optional[str]) -> Optional[FormulaGraph]:
    """OPT graph of a Content MathML string, or None if it is unparseable or empty."""
    root = parse(xml)
    if root is None:
        return None
    builder = _OPTBuilder()
    builder.node(root)
    return builder.g if builder.g.num_nodes else None


# ---------------------------------------------------------------------------
# Vocabulary and model input
# ---------------------------------------------------------------------------

PAD, UNK = "<pad>", "<unk>"
_NUMERIC_TAGS = {"mn", "cn"}
_INT_RE = re.compile(r"^\d+$")
_DEC_RE = re.compile(r"^\d*[.,]\d+$")


def number_bucket(symbol: str) -> str:
    if _INT_RE.match(symbol):
        return f"<int{min(len(symbol), 5)}>"
    if _DEC_RE.match(symbol):
        return "<decimal>"
    return "<number>"


_BUCKETS = [f"<int{i}>" for i in range(1, 6)] + ["<decimal>", "<number>"]


@dataclass
class GraphVocab:
    """Ids for tags, symbols and edge types of one representation ('slt' or 'opt')."""
    kind: str
    tags: Dict[str, int]
    symbols: Dict[str, int]
    edge_types: Dict[str, int]  # forward types; reverse type of t is t + len(edge_types)

    @property
    def num_edge_types(self) -> int:
        return 2 * len(self.edge_types)

    @classmethod
    def build(cls, kind: str, tag_counts: Counter, symbol_counts: Counter, min_count: int = 5) -> "GraphVocab":
        base = [PAD, UNK]
        tags = base + sorted(t for t, c in tag_counts.items() if c >= min_count)
        symbols = base + _BUCKETS + sorted(s for s, c in symbol_counts.items() if c >= min_count)
        edge_types = SLT_EDGE_TYPES if kind == "slt" else OPT_EDGE_TYPES
        return cls(kind, {t: i for i, t in enumerate(tags)}, {s: i for i, s in enumerate(symbols)},
                   {e: i for i, e in enumerate(edge_types)})

    def symbol_id(self, tag: str, symbol: str) -> int:
        if symbol in self.symbols:
            return self.symbols[symbol]
        if tag in _NUMERIC_TAGS and symbol:
            return self.symbols[number_bucket(symbol)]
        return self.symbols[UNK]

    def encode(self, graph: FormulaGraph, max_nodes: int = 0) -> Dict[str, np.ndarray]:
        """
        Arrays for a graph: node tag ids, node symbol ids, edge_index (2 × E, both
        directions) and edge types. Nodes are kept in construction order (a
        depth-first walk); with max_nodes > 0 the rest is cut off.
        """
        n = graph.num_nodes if max_nodes <= 0 else min(graph.num_nodes, max_nodes)
        unk = self.tags[UNK]
        tag_ids = np.array([self.tags.get(t, unk) for t in graph.tags[:n]], dtype=np.int64)
        sym_ids = np.array([self.symbol_id(t, s) for t, s in zip(graph.tags[:n], graph.symbols[:n])], dtype=np.int64)
        kept = [(a, b, self.edge_types[e]) for a, b, e in graph.edges if a < n and b < n]
        offset = len(self.edge_types)
        src = [a for a, _, _ in kept] + [b for _, b, _ in kept]
        dst = [b for _, b, _ in kept] + [a for a, _, _ in kept]
        etype = [t for _, _, t in kept] + [t + offset for _, _, t in kept]
        return {
            "tag": tag_ids,
            "symbol": sym_ids,
            "edge_index": np.array([src, dst], dtype=np.int64).reshape(2, -1),
            "edge_type": np.array(etype, dtype=np.int64),
            "truncated": np.array(n < graph.num_nodes),
        }

    def to_json(self) -> dict:
        return {"kind": self.kind, "tags": self.tags, "symbols": self.symbols, "edge_types": self.edge_types}

    @classmethod
    def from_json(cls, data: dict) -> "GraphVocab":
        return cls(data["kind"], data["tags"], data["symbols"], data["edge_types"])


def save_vocabs(vocabs: Dict[str, GraphVocab], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.to_json() for k, v in vocabs.items()}, ensure_ascii=False))


def load_vocabs(path: Path) -> Dict[str, GraphVocab]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data.graph_stats`")
    return {k: GraphVocab.from_json(v) for k, v in json.loads(path.read_text()).items()}


def to_pyg(graph: FormulaGraph, vocab: GraphVocab, max_nodes: int = 0):
    """PyTorch Geometric Data with x (N × 2: tag id, symbol id), edge_index and edge_type."""
    import torch
    from torch_geometric.data import Data

    arrays = vocab.encode(graph, max_nodes)
    x = torch.from_numpy(np.stack([arrays["tag"], arrays["symbol"]], axis=1))
    return Data(x=x, edge_index=torch.from_numpy(arrays["edge_index"]),
                edge_type=torch.from_numpy(arrays["edge_type"]), num_nodes=x.shape[0])
