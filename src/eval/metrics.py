"""
ARQMath Task 2 metrics, computed the way the official evaluation does.

Official protocol (data/raw/arqmath/eval_scripts/arqmath3/):
  1. runs are deduplicated by visual id (our runs are already visual-id level);
  2. at most 1000 results per topic are considered;
  3. "prime" filtering: visual ids without a judgment for the topic are removed;
  4. trec_eval computes nDCG′ with graded relevance, and MAP′ and P′@10 with
     `-l2` (grades 2–3 relevant).

Every topic in the qrels counts, and a topic with no judged results scores 0
(`missing_topics` lists them). This equals trec_eval with -c, which
src.eval.official uses; trec_eval without -c either skips such topics (older
versions) or refuses the run (current versions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pytrec_eval

from src.data.qrels import RELEVANT_GRADE, Qrels

Run = Dict[str, Dict[str, float]]  # topic_id → {visual_id: score}

MAX_DEPTH = 1000
MEASURES = ("ndcg", "map", "P_10")          # pytrec_eval result keys
LABELS = {"ndcg": "nDCG′", "map": "MAP′", "P_10": "P′@10", "judged_10": "judged@10"}


@dataclass
class EvalResult:
    mean: Dict[str, float]
    per_topic: Dict[str, Dict[str, float]]
    missing_topics: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{LABELS[m]} {self.mean[m]:.4f}" for m in (*MEASURES, "judged_10")]
        line = "  ".join(parts) + f"  ({len(self.per_topic)} topics)"
        if self.missing_topics:
            line += f"  [{len(self.missing_topics)} topics with no judged results, scored 0]"
        return line


def ranked(docs: Dict[str, float]) -> List[str]:
    """Documents in trec_eval order: score descending, ties by document id descending."""
    return [d for d, _ in sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def truncate(run: Run, depth: int = MAX_DEPTH) -> Run:
    """Top `depth` results per topic; depth <= 0 keeps everything (as the official scripts do)."""
    if depth <= 0:
        return run
    return {t: {d: docs[d] for d in ranked(docs)[:depth]} for t, docs in run.items()}


def prime(run: Run, qrels: Qrels) -> Run:
    """Keep only judged documents, for every topic in the qrels (absent topics become empty)."""
    return {t: {d: s for d, s in run.get(t, {}).items() if d in judged} for t, judged in qrels.items()}


def evaluate(run: Run, qrels: Qrels, depth: int = MAX_DEPTH) -> EvalResult:
    run = truncate(run, depth)
    primed = prime(run, qrels)
    scored = {t: docs for t, docs in primed.items() if docs}

    # Mirrors task2_get_results.py: nDCG without -l (graded gains), MAP and P with -l2.
    graded = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}, relevance_level=1).evaluate(scored)
    binary = pytrec_eval.RelevanceEvaluator(qrels, {"map", "P"}, relevance_level=RELEVANT_GRADE).evaluate(scored)

    per_topic: Dict[str, Dict[str, float]] = {}
    for t in qrels:
        top10 = ranked(run.get(t, {}))[:10]
        per_topic[t] = {
            "ndcg": graded.get(t, {}).get("ndcg", 0.0),
            "map": binary.get(t, {}).get("map", 0.0),
            "P_10": binary.get(t, {}).get("P_10", 0.0),
            # Share of the unfiltered top 10 that is judged: low values mean prime metrics
            # say little about how the system ranks unjudged formulas.
            "judged_10": sum(d in qrels[t] for d in top10) / 10,
        }

    n = len(per_topic)
    mean = {m: sum(v[m] for v in per_topic.values()) / n if n else 0.0 for m in (*MEASURES, "judged_10")}
    return EvalResult(mean=mean, per_topic=per_topic, missing_topics=sorted(t for t in qrels if t not in scored))
