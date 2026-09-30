"""
Fast retrieval check during training, on the dev topics (ARQMath-2).

Encoding all 8.4M formulas for a full evaluation takes too long to do every few
thousand steps. Instead, each dev topic's query formula is scored against a fixed
candidate set: every judged visual id of the dev qrels plus a fixed random sample of
corpus formulas as distractors. The top 1,000 candidates per topic are evaluated with
src.eval.metrics, so the numbers are nDCG′ / MAP′ / P′@10 / judged@10 as usual, but
computed over this reduced corpus: use them to compare checkpoints of one run, not as
final results (for those, run full retrieval on dev).

judged@10 is the most informative number here: distractors are unjudged, so it is the
share of the top 10 that the model fills with formulas the assessors saw (which were
pooled from strong systems) instead of random corpus formulas.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from src.data.formula_graph import GraphVocab, opt_graph, slt_graph
from src.data.qrels import Qrels, load_qrels
from src.data.topics import Topic, load_topics
from src.eval.metrics import MAX_DEPTH, EvalResult, evaluate
from src.model.encoder import FormulaEncoder, batch_to
from src.pretrain.dataset import collate_views, edge_type_counts
from src.pretrain.graph_store import REPS, GraphStore, arrays_from_graph


def topic_views(topics: Dict[str, Topic], vocabs: Dict[str, GraphVocab], max_nodes: int = 256) -> Dict[str, dict]:
    """Query views built from the topics' official SLT/OPT."""
    return {tid: {"slt": arrays_from_graph(slt_graph(t.slt), vocabs["slt"], max_nodes),
                  "opt": arrays_from_graph(opt_graph(t.opt), vocabs["opt"], max_nodes)}
            for tid, t in topics.items()}


@torch.no_grad()
def encode_views(model: FormulaEncoder, views: Sequence[dict], n_types: Dict[str, int], device,
                 batch_size: int = 1024) -> torch.Tensor:
    """[len(views), out_dim] on the CPU; views without any graph get the all-missing embedding."""
    out: List[torch.Tensor] = []
    for start in range(0, len(views), batch_size):
        batch = batch_to(collate_views(views[start:start + batch_size], n_types), device)
        out.append(model(batch).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, model.cfg.out_dim)


class QuickDev:
    def __init__(self, store: GraphStore, vocabs: Dict[str, GraphVocab], qrels: Qrels, topics: Dict[str, Topic],
                 n_distractors: int = 100_000, seed: int = 0, max_nodes: int = 256):
        self.store, self.qrels = store, qrels
        self.n_types = edge_type_counts(vocabs)
        judged = {vid for docs in qrels.values() for vid in docs}
        found = {i for i, vid in enumerate(store.visual_ids) if vid in judged}
        rng = np.random.default_rng(seed)
        distractors = rng.choice(len(store), size=min(n_distractors, len(store)), replace=False)
        self.candidates = sorted(found | set(int(i) for i in distractors))
        self.candidate_vids = [store.visual_ids[i] for i in self.candidates]
        self.judged_coverage = len(found) / len(judged) if judged else 0.0
        views = topic_views({t: topics[t] for t in qrels if t in topics}, vocabs, max_nodes)
        self.topic_ids = sorted(views)
        self.query_views = [views[t] for t in self.topic_ids]

    @classmethod
    def for_split(cls, store: GraphStore, vocabs: Dict[str, GraphVocab], split: str = "dev", **kwargs) -> "QuickDev":
        return cls(store, vocabs, load_qrels(split), load_topics(split), **kwargs)

    def run(self, model: FormulaEncoder, device, batch_size: int = 1024, depth: int = MAX_DEPTH) -> EvalResult:
        was_training = model.training
        model.eval()
        try:
            cand_views = [{rep: self.store.get(i, rep) for rep in REPS} for i in self.candidates]
            docs = encode_views(model, cand_views, self.n_types, device, batch_size)
            queries = encode_views(model, self.query_views, self.n_types, device, batch_size)
        finally:
            model.train(was_training)
        scores = queries @ docs.t()
        k = min(depth, scores.shape[1])
        top_scores, top_idx = scores.topk(k, dim=1)
        run = {tid: {self.candidate_vids[j]: float(s) for s, j in zip(top_scores[q].tolist(), top_idx[q].tolist())}
               for q, tid in enumerate(self.topic_ids)}
        return evaluate(run, self.qrels, depth=depth)
