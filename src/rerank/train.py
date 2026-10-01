"""
Train a linear reranker on judged training pairs and rerank a first-stage run.

Model: pairwise logistic regression (RankNet-style, linear). For each training topic,
every pair of judged candidates with different grades becomes one example: the feature
difference (better − worse), learned to be positive. Features are standardised; the
weights stay readable, so the model shows which kinds of match matter.

Reranking: for each topic the first-stage top-k is reordered by
    α · minmax(reranker score) + (1 − α) · minmax(first-stage score)
and the rest of the first-stage list follows unchanged. α = 1 is pure reranking;
α < 1 keeps some of the first stage's ranking (including the fine-tuned GNN, which
the reranker deliberately does not see as a feature).

--features picks the feature set: base (the 44 structural features + BM25 / GNN scores) or
all (+ the proto group, src.rerank.features), for the ablation; --apply uses the saved
model's own features.

The regularisation C, the depth k and α are chosen on the --dev features (ARQMath-2):
dev scores of the selected setting are therefore optimistic; apply it unchanged to test.
--cv k gives an honest dev estimate: topics are split into k folds, and each fold is
reranked with the setting chosen on the other folds (the model weights never see dev).
--cv-out writes that cross-validated run, for src.eval.compare against the first stage.

Usage
-----
    python -m src.rerank.train --train data/processed/rerank/train_judged.npz \\
        --dev data/processed/rerank/dev_rrf_ft.npz --dev-run runs/rrf_ft_pa05_dev.tsv --split dev \\
        --features all --cv 5 --cv-out runs/reranked_cv_dev.tsv
    python -m src.rerank.train --apply data/processed/rerank/model.json \\
        --dev data/processed/rerank/test_rrf_ft.npz --dev-run runs/rrf_ft_pa05_test.tsv --out runs/reranked_test.tsv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.paths import PROCESSED_DIR
from src.eval.metrics import Run, evaluate, ranked
from src.eval.runs import read_run, write_run
from src.finetune.train import load_train_qrels
from src.rerank.features import PROTO_FEATURE_NAMES

C_GRID = (0.001, 0.01, 0.1, 1.0)
ALPHA_GRID = tuple(round(a, 1) for a in np.linspace(0, 1, 11))
DEPTHS = (50, 100, 200, 300, 500)  # only those the features file covers are tried
FEATURE_SETS = ("base", "all")
MAX_PAIRS_PER_TOPIC = 5000


def load_features(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def feature_set(feat: dict, which: str) -> List[str]:
    names = [str(n) for n in feat["names"]]
    return names if which == "all" else [n for n in names if n not in set(PROTO_FEATURE_NAMES)]


def select_features(feat: dict, names: Sequence[str]) -> dict:
    """`feat` restricted to the named columns of X, in that order."""
    have = {str(n): i for i, n in enumerate(feat["names"])}
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(f"features file lacks {len(missing)} features, e.g. {missing[:3]}; rebuild it")
    return {**feat, "X": feat["X"][:, [have[n] for n in names]], "names": np.array(list(names))}


def pairwise_examples(feat: dict, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Difference vectors for judged pairs with different grades, half of them flipped."""
    rng = np.random.default_rng(seed)
    X, topics, grades = feat["X"], feat["topics"], feat["grades"]
    diffs: List[np.ndarray] = []
    for t in np.unique(topics):
        idx = np.flatnonzero((topics == t) & (grades >= 0))
        pairs = [(i, j) for i, j in itertools.combinations(idx, 2) if grades[i] != grades[j]]
        if len(pairs) > MAX_PAIRS_PER_TOPIC:  # keep topics with many judgments from dominating
            pairs = [pairs[k] for k in rng.choice(len(pairs), MAX_PAIRS_PER_TOPIC, replace=False)]
        for i, j in pairs:
            hi, lo = (i, j) if grades[i] > grades[j] else (j, i)
            diffs.append(X[hi] - X[lo])
    D = np.stack(diffs)
    y = np.ones(len(D), dtype=int)
    flip = rng.random(len(D)) < 0.5
    D[flip] *= -1
    y[flip] = 0
    return D, y


class LinearReranker:
    def __init__(self, names: Sequence[str], mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, **settings):
        self.names, self.mean, self.scale, self.weights = list(names), mean, scale, weights
        self.settings = settings

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X.astype(np.float64) - self.mean) / self.scale) @ self.weights

    def to_json(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "weights": self.weights.tolist(), "settings": self.settings}

    @classmethod
    def from_json(cls, d: dict) -> "LinearReranker":
        return cls(d["names"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["weights"]), **d["settings"])


def fit(train: dict, C: float, seed: int = 0) -> LinearReranker:
    from sklearn.linear_model import LogisticRegression

    X = train["X"].astype(np.float64)  # float64 throughout, so a saved and reloaded model scores identically
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    D, y = pairwise_examples({**train, "X": (X - mean) / scale}, seed)
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(D, y)
    return LinearReranker(train["names"], mean, scale, clf.coef_.ravel(), C=C)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = (x.min(), x.max()) if len(x) else (0.0, 0.0)
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


def rerank(first_stage: Run, feat: dict, scores: np.ndarray, depth: int, alpha: float) -> Run:
    """Reorder each topic's first-stage top-`depth`; candidates without features keep their order below."""
    by_topic: Dict[str, Dict[str, float]] = {}
    for t, v, s in zip(feat["topics"], feat["visual_ids"], scores):
        by_topic.setdefault(str(t), {})[str(v)] = float(s)
    out: Run = {}
    for t, docs in first_stage.items():
        order = ranked(docs)
        top = [v for v in order[:depth] if v in by_topic.get(t, {})]
        in_top = set(top)
        rest = [v for v in order if v not in in_top]
        if top:
            r = _minmax(np.array([by_topic[t][v] for v in top]))
            f = _minmax(np.array([docs[v] for v in top]))
            combined = alpha * r + (1 - alpha) * f
            top = [v for _, v in sorted(zip(combined, top), key=lambda x: -x[0])]
        new_order = top + rest
        out[t] = {v: float(len(new_order) - k) for k, v in enumerate(new_order)}
    return out


Setting = Tuple[float, int, float]          # C, depth, α
PerTopic = Dict[str, Dict[str, float]]      # topic → measure → value


def _mean(per_topic: PerTopic, topics: Sequence[str]) -> Dict[str, float]:
    measures = next(iter(per_topic.values())).keys()
    return {m: sum(per_topic[t][m] for t in topics) / len(topics) for m in measures}


def select(results: Dict[Setting, PerTopic], topics: Sequence[str]) -> Setting:
    """Best setting by mean (nDCG′, MAP′) over `topics`; ties → the earliest in grid order."""
    def key(s: Setting) -> Tuple[float, float]:
        m = _mean(results[s], topics)
        return m["ndcg"], m["map"]
    return max(results, key=key)


def cross_validate(results: Dict[Setting, PerTopic], scores: Dict[float, np.ndarray], first_stage: Run,
                   feat: dict, topics: Sequence[str], folds: int, seed: int = 0) -> Tuple[Run, List[Setting]]:
    """Run whose held-out topics are reranked with the setting chosen on the other folds."""
    topics = sorted(topics)
    order = np.random.default_rng(seed).permutation(len(topics))  # same folds as src.eval.fuse --cv
    fold_of = {topics[i]: k % folds for k, i in enumerate(order)}
    run: Run = {}
    chosen: List[Setting] = []
    for f in range(folds):
        setting = select(results, [t for t in topics if fold_of[t] != f])
        chosen.append(setting)
        C, depth, alpha = setting
        full = rerank(first_stage, feat, scores[C], depth, alpha)
        run.update({t: full[t] for t in topics if fold_of[t] == f and t in full})
    return run, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, help="judged training features (src.rerank.build --judged)")
    parser.add_argument("--apply", type=Path, help="apply a saved model.json instead of training")
    parser.add_argument("--dev", type=Path, required=True, help="features of the first-stage candidates to rerank")
    parser.add_argument("--dev-run", type=Path, required=True, help="the first-stage run those candidates came from")
    parser.add_argument("--split", help="split whose qrels select C / depth / α (training mode)")
    parser.add_argument("--qrels", default="official", help="official / all / path to a qrels file")
    parser.add_argument("--model-out", type=Path, default=PROCESSED_DIR / "rerank/model.json")
    parser.add_argument("--out", type=Path, help="reranked run file")
    parser.add_argument("--run-id", default="reranked")
    parser.add_argument("--features", choices=FEATURE_SETS, default="all", help="feature set (training mode)")
    parser.add_argument("--cv", type=int, default=0, help="k-fold cross-validated dev estimate (training mode)")
    parser.add_argument("--cv-out", type=Path, help="write the cross-validated run here")
    args = parser.parse_args()
    if (args.cv or args.cv_out) and args.apply:
        parser.error("--cv needs training mode, not --apply")
    if args.cv_out and not args.cv:
        parser.error("--cv-out needs --cv")

    dev = load_features(args.dev)
    first_stage, _ = read_run(args.dev_run)

    if args.apply:
        model = LinearReranker.from_json(json.loads(args.apply.read_text()))
        dev = select_features(dev, model.names)
        s = model.settings
        result_run = rerank(first_stage, dev, model.score(dev["X"]), s["depth"], s["alpha"])
        print(f"applied {args.apply}: C={s['C']}, depth={s['depth']}, α={s['alpha']}")
    else:
        if args.train is None or args.split is None:
            parser.error("training needs --train and --split")
        names = feature_set(dev, args.features)
        train, dev = select_features(load_features(args.train), names), select_features(dev, names)
        qrels = load_train_qrels(args.qrels, args.split)
        covered = int(np.unique(dev["topics"], return_counts=True)[1].max())
        depths = [d for d in DEPTHS if d <= covered] or [covered]
        print(f"features: {args.features} ({len(names)}); depths {depths} (features cover the top {covered})")
        print(f"first stage on {args.split}: {evaluate(first_stage, qrels).summary()}")
        topics = list(qrels)
        models: Dict[float, LinearReranker] = {}
        scores: Dict[float, np.ndarray] = {}
        results: Dict[Setting, PerTopic] = {}
        for C in C_GRID:
            models[C] = fit(train, C)
            scores[C] = models[C].score(dev["X"])
            for depth in depths:
                row = {}
                for a in ALPHA_GRID:
                    row[(C, depth, a)] = evaluate(rerank(first_stage, dev, scores[C], depth, a), qrels).per_topic
                results.update(row)
                best_a = select(row, topics)
                pure, m = _mean(row[(C, depth, ALPHA_GRID[-1])], topics), _mean(row[best_a], topics)
                print(f"  C={C:<6} depth={depth:<4} rerank only: nDCG′ {pure['ndcg']:.4f} MAP′ {pure['map']:.4f}"
                      f" | best α={best_a[2]:.1f}: nDCG′ {m['ndcg']:.4f} MAP′ {m['map']:.4f} P′@10 {m['P_10']:.4f}")
        C, depth, alpha = select(results, topics)
        model = models[C]
        model.settings.update(depth=depth, alpha=alpha)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_text(json.dumps(model.to_json(), indent=2))
        print(f"\nselected C={C}, depth={depth}, α={alpha} (chosen on {args.split}: optimistic)"
              f" → {args.model_out}")
        top = sorted(zip(model.weights, model.names), key=lambda x: -abs(x[0]))[:12]
        print("largest weights (standardised features): " + ", ".join(f"{n} {w:+.2f}" for w, n in top))
        result_run = rerank(first_stage, dev, scores[C], depth, alpha)
        print(f"reranked on {args.split}: {evaluate(result_run, qrels).summary()}")

        if args.cv:
            cv_run, chosen = cross_validate(results, scores, first_stage, dev, topics, args.cv)
            print(f"\n{args.cv}-fold cross-validated on {args.split}: {evaluate(cv_run, qrels).summary()}")
            print("fold settings (C, depth, α): " + ", ".join(f"({c}, {d}, {a})" for c, d, a in chosen))
            if args.cv_out:
                write_run(cv_run, args.cv_out, f"{args.run_id}_cv")
                print(f"wrote {args.cv_out}")

    if args.out:
        write_run(result_run, args.out, args.run_id)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
