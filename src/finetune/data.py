"""
Supervised fine-tuning data from the ARQMath relevance judgments.

For every training topic (ARQMath-1 by default, all judgments) the query is the topic's
official formula, positives are its visual ids graded 2–3, and hard negatives are its
visual ids that the assessors judged 0–1: formulas that looked plausible enough to be
pooled, but are not relevant. Unjudged formulas are never used as negatives.

Batches are built from distinct topics (TopicBatchSampler): each batch draws B different
topics and, per topic, one positive and one hard negative. With one example per topic,
no other item in the batch can be a hidden positive for a query, so no masking is
needed, and topics with many positives do not dominate training. For query i the
candidates are all B positives and all B hard negatives; the right answer is positive i.

Topics whose judged formulas include no hard negative get a random corpus formula instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.formula_graph import GraphVocab
from src.data.qrels import RELEVANT_GRADE, Qrels
from src.data.topics import Topic
from src.model.quick_dev import topic_views
from src.pretrain.augment import View
from src.pretrain.dataset import collate_views, edge_type_counts
from src.pretrain.graph_store import REPS, GraphStore


@dataclass
class TopicExamples:
    topic_id: str
    query: View
    positives: List[int]   # store indices of grade ≥ 2 visual ids
    negatives: List[int]   # store indices of judged grade 0–1 visual ids


def build_examples(store: GraphStore, vocabs: Dict[str, GraphVocab], qrels: Qrels, topics: Dict[str, Topic],
                   max_nodes: int = 256) -> Tuple[List[TopicExamples], dict]:
    judged = {vid for docs in qrels.values() for vid in docs}
    index = {vid: i for i, vid in enumerate(store.visual_ids) if vid in judged}
    views = topic_views({t: topics[t] for t in qrels if t in topics}, vocabs, max_nodes)
    examples, skipped = [], {"no_topic": 0, "no_query_graph": 0, "no_positive": 0}
    for tid in sorted(qrels):
        if tid not in views:
            skipped["no_topic"] += 1
            continue
        if all(views[tid][rep] is None for rep in REPS):
            skipped["no_query_graph"] += 1
            continue
        pos = sorted(index[v] for v, g in qrels[tid].items() if g >= RELEVANT_GRADE and v in index)
        neg = sorted(index[v] for v, g in qrels[tid].items() if g < RELEVANT_GRADE and v in index)
        if not pos:
            skipped["no_positive"] += 1
            continue
        examples.append(TopicExamples(tid, views[tid], pos, neg))
    stats = {
        "topics": len(qrels),
        "used_topics": len(examples),
        "skipped": skipped,
        "positives": sum(len(e.positives) for e in examples),
        "negatives": sum(len(e.negatives) for e in examples),
        "topics_without_negatives": sum(not e.negatives for e in examples),
        "judged_in_store": len(index) / len(judged) if judged else 0.0,
    }
    return examples, stats


class FinetuneTriples:
    """
    Map-style over topics: item k is (anchor, positive, hard negative) for topic k.

    The anchor is the topic's query formula, or, with probability p_positive_anchor
    and when the topic has at least two relevant formulas, one of its relevant
    formulas paired with a different relevant one: two formulas judged relevant to
    the same query are usually close to each other, and this turns 74 query formulas
    into hundreds of distinct anchors.
    """

    def __init__(self, store: GraphStore, examples: Sequence[TopicExamples], seed: int = 0,
                 p_positive_anchor: float = 0.0):
        self.store, self.examples, self.seed = store, list(examples), seed
        self.p_positive_anchor = p_positive_anchor
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def _view(self, i: int) -> View:
        return {rep: self.store.get(i, rep) for rep in REPS}

    def __getitem__(self, key) -> Tuple[View, View, View]:
        k, draw = key  # draw: which sampling round, so a topic drawn in many batches gets different pairs
        ex = self.examples[k]
        rng = np.random.default_rng((self.seed, self.epoch, k, draw))
        neg = int(rng.choice(ex.negatives)) if ex.negatives else int(rng.integers(len(self.store)))
        if len(ex.positives) >= 2 and rng.random() < self.p_positive_anchor:
            anchor, pos = (int(i) for i in rng.choice(ex.positives, size=2, replace=False))
            return self._view(anchor), self._view(pos), self._view(neg)
        pos = int(rng.choice(ex.positives))
        return ex.query, self._view(pos), self._view(neg)


class TopicBatchSampler:
    """Yields `steps` batches, each of `batch_topics` distinct topics, as (topic, draw) keys."""

    def __init__(self, n_topics: int, batch_topics: int, steps: int, seed: int = 0):
        self.n, self.b, self.steps, self.seed = n_topics, min(batch_topics, n_topics), steps, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for draw in range(self.steps):
            yield [(int(k), draw) for k in rng.choice(self.n, size=self.b, replace=False)]


class TripleCollator:
    """(query, positive, negative) triples → {"query": batched queries, "docs": batched positives then negatives}."""

    def __init__(self, vocabs: Dict[str, GraphVocab]):
        self.n_types = edge_type_counts(vocabs)

    def __call__(self, batch: Sequence[Tuple[View, View, View]]) -> dict:
        queries = [t[0] for t in batch]
        docs = [t[1] for t in batch] + [t[2] for t in batch]
        return {"query": collate_views(queries, self.n_types), "docs": collate_views(docs, self.n_types)}
