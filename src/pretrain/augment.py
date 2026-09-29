"""
Augmentations that turn one formula into a training view for contrastive pretraining.

A view is {"slt": arrays or None, "opt": arrays or None}, where arrays are the
per-formula dicts of GraphStore.get (tag, symbol, src, dst, type). Two independent
views of the same formula form a positive pair.

  rename     each distinct variable is replaced by another variable of the same kind
             (lower/upper Latin, lower/upper Greek), consistently across the formula and
             across its SLT and OPT graphs: x²−y ↦ a²−b. Constants e, i, π are kept.
  numbers    each distinct integer other than 0 and 1 is replaced by another integer with
             the same number of digits, consistently across SLT and OPT.
  crop       a random subtree (a node and everything reachable from it) covering at most
             max_crop of the graph is removed, independently in each graph; in SLT this
             cuts a script, a fraction part or the end of a baseline.
  modality   one of the two graphs is dropped, so each encoder branch also learns to
             represent the formula alone (and the model copes with the ~0.3% of formulas
             that only have one graph).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.formula_graph import GraphVocab

Arrays = Dict[str, np.ndarray]
View = Dict[str, Optional[Arrays]]

_IDENT_TAGS = {"slt": ("mi",), "opt": ("ci", "csymbol")}
_NUMBER_TAGS = {"slt": ("mn",), "opt": ("cn",)}
_CONSTANTS = {"e", "i", "π"}
_FIXED_NUMBERS = {"0", "1"}


def variable_class(symbol: str) -> Optional[str]:
    if len(symbol) != 1 or symbol in _CONSTANTS:
        return None
    if "a" <= symbol <= "z":
        return "latin_lower"
    if "A" <= symbol <= "Z":
        return "latin_upper"
    if "α" <= symbol <= "ω" and symbol != "ς":
        return "greek_lower"
    if "Α" <= symbol <= "Ω":
        return "greek_upper"
    return None


def number_class(symbol: str) -> Optional[str]:
    if symbol.isdigit() and symbol.isascii() and symbol not in _FIXED_NUMBERS:
        return f"int{len(symbol)}"
    return None


@dataclass
class AugmentConfig:
    p_rename: float = 0.8
    p_numbers: float = 0.3
    p_crop: float = 0.3
    max_crop: float = 0.4          # largest share of a graph's nodes one crop may remove
    min_crop_nodes: int = 4        # smaller graphs are never cropped
    p_drop_modality: float = 0.15


class Augmenter:
    """Builds symbol-substitution tables once from the vocabularies, then augments views."""

    def __init__(self, vocabs: Dict[str, GraphVocab], config: AugmentConfig = AugmentConfig()):
        self.config = config
        self.reps = ("slt", "opt")
        # Symbols eligible for substitution, keyed by string so SLT and OPT stay consistent.
        # Only strings present in both vocabularies can be substituted in.
        self.pools: Dict[str, List[str]] = {}
        # Per representation: symbol id → symbol string, for ids that may be substituted.
        self.substitutable: Dict[str, Dict[int, str]] = {}
        self.ids: Dict[str, Dict[str, int]] = {rep: vocabs[rep].symbols for rep in self.reps}
        self.tag_ids: Dict[str, Dict[str, set]] = {}
        shared = set(vocabs["slt"].symbols) & set(vocabs["opt"].symbols)
        for symbol in sorted(shared):
            cls = variable_class(symbol) or number_class(symbol)
            if cls:
                self.pools.setdefault(cls, []).append(symbol)
        for rep in self.reps:
            self.substitutable[rep] = {i: s for s, i in vocabs[rep].symbols.items()
                                       if variable_class(s) or number_class(s)}
            tags = vocabs[rep].tags
            self.tag_ids[rep] = {
                "ident": {tags[t] for t in _IDENT_TAGS[rep] if t in tags},
                "number": {tags[t] for t in _NUMBER_TAGS[rep] if t in tags},
            }

    # -- symbol substitution --------------------------------------------------

    def _present(self, view: View, kind: str) -> List[str]:
        """Distinct substitutable symbols of one kind ('ident' or 'number') in the view."""
        found = set()
        for rep, g in view.items():
            if g is None:
                continue
            mask = np.isin(g["tag"], list(self.tag_ids[rep][kind]))
            for sid in np.unique(g["symbol"][mask]):
                s = self.substitutable[rep].get(int(sid))
                if s is not None and ((variable_class(s) is not None) == (kind == "ident")):
                    found.add(s)
        return sorted(found)

    def _substitute(self, view: View, kind: str, rng: np.random.Generator) -> View:
        symbols = self._present(view, kind)
        if not symbols:
            return view
        classify = variable_class if kind == "ident" else number_class
        mapping: Dict[str, str] = {}
        for cls in sorted({classify(s) for s in symbols}):
            members = [s for s in symbols if classify(s) == cls]
            pool = self.pools.get(cls, [])
            if len(pool) < len(members):
                continue
            chosen = rng.choice(len(pool), size=len(members), replace=False)
            mapping.update(zip(members, (pool[j] for j in chosen)))
        out: View = {}
        for rep, g in view.items():
            if g is None:
                out[rep] = None
                continue
            ids = self.ids[rep]
            table = {ids[old]: ids[new] for old, new in mapping.items() if old in ids and new in ids}
            eligible = np.isin(g["tag"], list(self.tag_ids[rep][kind]))
            symbol = g["symbol"].copy()
            for k in np.flatnonzero(eligible):
                symbol[k] = table.get(int(symbol[k]), symbol[k])
            out[rep] = {**g, "symbol": symbol}
        return out

    # -- structure ------------------------------------------------------------

    def _crop(self, g: Arrays, rng: np.random.Generator) -> Arrays:
        n = len(g["tag"])
        if n < self.config.min_crop_nodes:
            return g
        children: List[List[int]] = [[] for _ in range(n)]
        for a, b in zip(g["src"], g["dst"]):
            children[a].append(b)
        for _ in range(5):  # a few tries to find a subtree that is not too large
            root = int(rng.integers(1, n))
            drop, stack = {root}, [root]
            while stack:
                for c in children[stack.pop()]:
                    if c not in drop:
                        drop.add(c)
                        stack.append(c)
            if len(drop) <= self.config.max_crop * n:
                keep = np.array([k not in drop for k in range(n)])
                new_index = np.cumsum(keep) - 1
                edge_keep = keep[g["src"]] & keep[g["dst"]]
                return {"tag": g["tag"][keep], "symbol": g["symbol"][keep],
                        "src": new_index[g["src"][edge_keep]], "dst": new_index[g["dst"][edge_keep]],
                        "type": g["type"][edge_keep]}
        return g

    # -- views -----------------------------------------------------------------

    def view(self, graphs: View, rng: np.random.Generator) -> View:
        c = self.config
        v: View = dict(graphs)
        if rng.random() < c.p_rename:
            v = self._substitute(v, "ident", rng)
        if rng.random() < c.p_numbers:
            v = self._substitute(v, "number", rng)
        for rep in self.reps:
            if v[rep] is not None and rng.random() < c.p_crop:
                v[rep] = self._crop(v[rep], rng)
        present = [rep for rep in self.reps if v[rep] is not None]
        if len(present) == 2 and rng.random() < c.p_drop_modality:
            v[present[int(rng.integers(2))]] = None
        return v

    def pair(self, graphs: View, rng: np.random.Generator) -> Tuple[View, View]:
        return self.view(graphs, rng), self.view(graphs, rng)
