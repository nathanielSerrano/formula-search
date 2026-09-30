"""
Contrastive pretraining pairs: two augmented views of the same corpus formula.

    store = GraphStore()
    vocabs = load_vocabs(PROCESSED_DIR / "graph_vocab.json")
    data = PretrainPairs(store, Augmenter(vocabs))
    loader = DataLoader(data, batch_size=512, shuffle=True, num_workers=8,
                        collate_fn=PairCollator(vocabs))

Each batch holds, for view "a" and view "b", one PyG Batch per representation plus the
positions of the formulas that have that graph (collate_views) (the others were dropped by augmentation
or have no graph at all). Reverse edges are added here, with type t + T for a forward
type t out of T, as in GraphVocab.encode.

Formulas with fewer than `min_nodes` nodes in both graphs (single symbols such as x)
are not used as anchors: after renaming they are indistinguishable from one another,
so they would mostly act as false negatives within a batch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.formula_graph import GraphVocab
from src.pretrain.augment import Augmenter, View
from src.pretrain.graph_store import REPS, GraphStore


class PretrainPairs:
    """Map-style dataset (usable with torch DataLoader); item k is a pair of views of anchor k."""

    def __init__(self, store: GraphStore, augmenter: Augmenter, min_nodes: int = 3, seed: int = 0):
        self.store = store
        self.augmenter = augmenter
        self.seed = seed
        self.epoch = 0
        big_enough = (store.num_nodes("slt") >= min_nodes) | (store.num_nodes("opt") >= min_nodes)
        self.anchors = np.flatnonzero(big_enough)

    def set_epoch(self, epoch: int) -> None:
        """Different augmentations each epoch, reproducible for a given (seed, epoch, item)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, k: int) -> Tuple[View, View]:
        i = int(self.anchors[k])
        rng = np.random.default_rng((self.seed, self.epoch, i))
        graphs = {rep: self.store.get(i, rep) for rep in REPS}
        return self.augmenter.pair(graphs, rng)


def with_reverse_edges(g: Dict[str, np.ndarray], n_types: int) -> Tuple[np.ndarray, np.ndarray]:
    """(edge_index 2 × 2E, edge_type 2E): forward edges, then the same edges reversed."""
    edge_index = np.stack([np.concatenate([g["src"], g["dst"]]), np.concatenate([g["dst"], g["src"]])])
    edge_type = np.concatenate([g["type"], g["type"] + n_types])
    return edge_index.astype(np.int64), edge_type.astype(np.int64)


def collate_views(views: Sequence[View], n_types: Dict[str, int]) -> dict:
    """
    Batch a list of views for the encoder:
      {"size": B, "slt": {"graphs": Batch, "present": LongTensor} or None, "opt": …}
    `present` lists the positions (0..B-1) of the views that have that graph.
    """
    import torch
    from torch_geometric.data import Batch, Data

    out: dict = {"size": len(views)}
    for rep in REPS:
        data: List = []
        present: List[int] = []
        for pos, view in enumerate(views):
            g = view.get(rep)
            if g is None or len(g["tag"]) == 0:
                continue
            edge_index, edge_type = with_reverse_edges(g, n_types[rep])
            x = torch.from_numpy(np.stack([g["tag"], g["symbol"]], axis=1).astype(np.int64))
            data.append(Data(x=x, edge_index=torch.from_numpy(edge_index),
                             edge_type=torch.from_numpy(edge_type), num_nodes=x.shape[0]))
            present.append(pos)
        out[rep] = ({"graphs": Batch.from_data_list(data), "present": torch.tensor(present, dtype=torch.long)}
                    if data else None)
    return out


def edge_type_counts(vocabs: Dict[str, GraphVocab]) -> Dict[str, int]:
    """Number of forward edge types per representation (reverse types are offset by this)."""
    return {rep: len(vocabs[rep].edge_types) for rep in REPS}


class PairCollator:
    """Turns a list of (view_a, view_b) into {"a": batched views, "b": batched views} (see collate_views)."""

    def __init__(self, vocabs: Dict[str, GraphVocab]):
        self.n_types = edge_type_counts(vocabs)

    def __call__(self, batch: Sequence[Tuple[View, View]]) -> dict:
        return {"a": collate_views([p[0] for p in batch], self.n_types),
                "b": collate_views([p[1] for p in batch], self.n_types)}
