"""
Draw the system figures as standalone SVG files (no dependencies beyond the standard library).

    python docs/figures/make_figures.py          # writes docs/figures/*.svg

  pipeline.svg   whole system: offline build and training, online retrieval per query
  encoder.svg    dual-branch graph encoder (SLT + OPT → one 256-d vector)
  training.svg   contrastive pretraining and supervised fine-tuning
  reranker.svg   structural reranker: features, linear model, training on judged pairs
  protocol.svg   experimental protocol: train / dev / final retrain / test once

For a paper, convert with e.g. `rsvg-convert -f pdf -o pipeline.pdf pipeline.svg` or Inkscape.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent
FONT = "Helvetica, Arial, sans-serif"
INK, MUTED, ARROW, PANEL = "#1F2937", "#4B5563", "#4B5563", "#FAFAFB"
KINDS = {  # fill, stroke
    "data": ("#F3F4F6", "#6B7280"),
    "sparse": ("#E8F1FB", "#2B6CB0"),
    "neural": ("#F1ECFB", "#6B46C1"),
    "fuse": ("#FDF0E6", "#C05621"),
    "judged": ("#E8F6EE", "#2F855A"),
}
Point = Tuple[float, float]


class Figure:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.parts: List[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, size: float = 12, weight: str = "normal", anchor: str = "middle",
             color: str = INK, italic: bool = False, spacing: float = 0) -> None:
        style = ' font-style="italic"' if italic else ""
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
                 f'fill="{color}"{style}{ls}>{escape(s)}</text>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str, rx: float = 8,
             dashed: bool = False, width: float = 1.5) -> None:
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"{dash}/>')

    def box(self, x: float, y: float, w: float, h: float, kind: str, title: str, lines: Sequence[str] = (),
            title_size: float = 13.5, line_size: float = 11.5, dashed: bool = False) -> None:
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke, dashed=dashed)
        n = 1 + len(lines)
        block = title_size + (len(lines) * (line_size + 5))
        top = y + (h - block) / 2 + title_size - 2
        self.text(x + w / 2, top, title, size=title_size, weight="bold")
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + (i + 1) * (line_size + 5) + 1, line, size=line_size, color=MUTED)

    def panel(self, x: float, y: float, w: float, h: float, label: str) -> None:
        self.rect(x, y, w, h, PANEL, "#D1D5DB", rx=12, width=1)
        self.text(x + 16, y + 24, label.upper(), size=11, weight="bold", anchor="start", color=MUTED, spacing=1.2)

    def path(self, pts: Sequence[Point], color: str = ARROW, dashed: bool = False, arrow: bool = True,
             width: float = 1.6) -> None:
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        marker = f' marker-end="url(#{self._marker(color)})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{marker} '
                 f'stroke-linejoin="round"/>')

    def label(self, x: float, y: float, s: str, color: str = MUTED, size: float = 11, anchor: str = "middle") -> None:
        width = len(s) * size * 0.56 + 10
        left = x - width / 2 if anchor == "middle" else x - 5
        self.add(f'<rect x="{left}" y="{y - size}" width="{width}" height="{size + 6}" rx="3" fill="white"/>')
        self.text(x, y, s, size=size, color=color, italic=True, anchor=anchor)

    _markers: dict

    def _marker(self, color: str) -> str:
        if not hasattr(self, "_markers"):
            self._markers = {}
        if color not in self._markers:
            self._markers[color] = f"arrow{len(self._markers)}"
        return self._markers[color]

    def node(self, x: float, y: float, s: str, kind: str = "neural", r: float = 13) -> None:
        fill, stroke = KINDS[kind]
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{stroke}" stroke-width="1.4"/>')
        self.text(x, y + 4.5, s, size=12.5, color=INK)

    def edge(self, a: Point, b: Point, s: str = "", at: Tuple[float, float, str] = None, r: float = 13,
             color: str = "#6B7280") -> None:
        """Arrow between two nodes; `at` = (x, y, anchor) of the edge label (default: above the midpoint)."""
        (x1, y1), (x2, y2) = a, b
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** 0.5
        p1 = (x1 + dx / n * r, y1 + dy / n * r)
        p2 = (x2 - dx / n * (r + 3), y2 - dy / n * (r + 3))
        self.path([p1, p2], color=color, width=1.2)
        if s:
            x, y, anchor = at or ((x1 + x2) / 2, (y1 + y2) / 2 - 6, "middle")
            self.text(x, y, s, size=9.5, color=MUTED, italic=True, anchor=anchor)

    def save(self, name: str) -> Path:
        markers = "".join(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            for color, mid in getattr(self, "_markers", {}).items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
               f'<defs>{markers}</defs><rect width="100%" height="100%" fill="white"/>'
               + "".join(self.parts) + "</svg>\n")
        path = OUT_DIR / name
        path.write_text(svg, encoding="utf-8")
        return path

    def legend(self, x: float, y: float, items: Sequence[Tuple[str, str]]) -> None:
        for kind, s in items:
            fill, stroke = KINDS[kind]
            self.rect(x, y - 10, 14, 14, fill, stroke, rx=3, width=1.2)
            self.text(x + 20, y + 1, s, size=11, anchor="start", color=MUTED)
            x += 28 + len(s) * 6.2


def pipeline() -> Path:
    f = Figure(1180, 880)
    L, R, W, SW = 40, 640, 440, 210          # column lefts, full width, sub-box width
    rows = {1: 80, 2: 175, 3: 270, 4: 365, 5: 470, 6: 575, 7: 680}
    H = 66
    f.panel(20, 40, 480, 760, "Offline · run once")
    f.panel(620, 40, 480, 760, "Online · per query formula")

    # Offline column
    f.box(L, rows[1], W, H, "data", "ARQMath collection",
          ["28.3M formula instances from Math Stack Exchange, 2010–2018", "LaTeX, SLT and OPT (MathML via LaTeXML)"])
    f.box(L, rows[2], W, H, "data", "Formula index",
          ["MathML sanitised (unclosed tags, raw & and <)", "issue flags: deleted formulas never returned"])
    f.box(L, rows[3], W, H, "data", "Visual index",
          ["one representative per visually distinct formula", "8.4M retrievable visual IDs"])
    f.box(L, rows[4], SW, H, "data", "Graph store", ["SLT + OPT graphs, typed edges", "≤256 nodes, 20k-symbol vocab"])
    f.box(L + 230, rows[4], SW, H, "sparse", "BM25 index", ["SLT symbol unigrams + bigrams", "k1 = 1.2, b = 0.75"])
    f.box(L, rows[5], SW, H, "neural", "Contrastive pretraining", ["8.4M formulas, 2 views each", "InfoNCE, in-batch negatives"])
    f.box(L + 230, rows[5], SW, H, "judged", "Relevance judgments", ["ARQMath-1 (development)", "ARQMath-1 + 2 (final models)"])
    f.box(L, rows[6], SW, H, "neural", "Supervised fine-tuning", ["judged positives, hard negatives", "then encode all 8.4M (256-d)"])
    f.box(L + 230, rows[6], SW, H, "fuse", "Reranker training", ["judged pairs, 48 features", "pairwise logistic regression"])

    cx, lc, rc = L + W / 2, L + SW / 2, L + 230 + SW / 2
    for a, b in ((1, 2), (2, 3)):
        f.path([(cx, rows[a] + H), (cx, rows[b])])
    split = rows[3] + H + 14
    f.path([(cx, rows[3] + H), (cx, split), (lc, split), (lc, rows[4])])
    f.path([(cx, split), (rc, split), (rc, rows[4])])
    f.path([(lc, rows[4] + H), (lc, rows[5])])
    f.path([(lc - 30, rows[5] + H), (lc - 30, rows[6])])
    mid = rows[5] + H + 19
    f.path([(rc - 50, rows[5] + H), (rc - 50, mid), (lc + 50, mid), (lc + 50, rows[6])], color=KINDS["judged"][1])
    f.path([(rc + 40, rows[5] + H), (rc + 40, rows[6])], color=KINDS["judged"][1])

    # Online column
    f.box(R, rows[3], W, H, "judged", "Topic formula",
          ["official SLT + OPT of the query formula", "dev: ARQMath-2 · test: ARQMath-3"])
    f.box(R, rows[4], SW, H, "sparse", "BM25 search", ["sparse retrieval", "top 1,000"])
    f.box(R + 230, rows[4], SW, H, "neural", "Dense GNN search", ["exact inner product", "top 1,000"])
    f.box(R, rows[5], W, H, "fuse", "Reciprocal rank fusion", ["score = Σ 1 / (60 + rank)"])
    f.box(R, rows[6], W, H, "fuse", "Structural reranker",
          ["reorders the top 300: symbol pairs, paths, tree edit distance,", "BM25 and pretrained-GNN scores"])
    f.box(R, rows[7], W, H, "judged", "Ranked list → evaluation",
          ["deduplicated by visual ID; prime nDCG′, MAP′, P′@10", "paired randomization tests"])

    rcx, rl, rr = R + W / 2, R + SW / 2, R + 230 + SW / 2
    split = rows[3] + H + 14
    f.path([(rcx, rows[3] + H), (rcx, split), (rl, split), (rl, rows[4])])
    f.path([(rcx, split), (rr, split), (rr, rows[4])])
    merge = rows[4] + H + 19
    f.path([(rl, rows[4] + H), (rl, merge), (rcx, merge)], arrow=False)
    f.path([(rr, rows[4] + H), (rr, merge), (rcx, merge), (rcx, rows[5])])
    for a, b in ((5, 6), (6, 7)):
        f.path([(rcx, rows[a] + H), (rcx, rows[b])])

    # What the offline side hands to the online side
    y4 = rows[4] + H / 2
    f.path([(L + W, y4), (R, y4)], color=KINDS["sparse"][1], dashed=True)
    f.label((L + W + R) / 2, y4 - 7, "index", color=KINDS["sparse"][1])
    y6 = rows[6] + H / 2
    f.path([(L + W, y6), (R, y6)], color=KINDS["fuse"][1], dashed=True)
    f.label((L + W + R) / 2, y6 - 7, "trained model", color=KINDS["fuse"][1])
    bottom, right = 822, 1135
    f.path([(lc, rows[6] + H), (lc, bottom), (right, bottom), (right, y4), (R + W, y4)],
           color=KINDS["neural"][1], dashed=True)
    f.label(560, bottom + 4, "fine-tuned encoder + 8.4M formula vectors", color=KINDS["neural"][1])

    f.legend(40, 862, [("data", "data"), ("sparse", "sparse retrieval"), ("neural", "graph neural network"),
                       ("fuse", "fusion and reranking"), ("judged", "topics, judgments, evaluation")])
    return f.save("pipeline.svg")


def encoder() -> Path:
    f = Figure(1180, 440)
    f.text(30, 30, "One encoder branch per representation (separate weights); both branches meet in the fusion MLP.",
           size=12, anchor="start", color=MUTED)
    rows = {"slt": 55, "opt": 245}
    H = 150
    # Example graphs for x² − y
    f.rect(30, rows["slt"], 220, H, "white", KINDS["neural"][1])
    f.text(140, rows["slt"] + 22, "SLT (appearance)", size=13, weight="bold")
    y = rows["slt"] + 100
    f.node(60, y, "x"), f.node(95, y - 45, "2"), f.node(140, y, "−"), f.node(205, y, "y")
    f.edge((60, y), (95, y - 45), "sup", at=(70, y - 28, "end"))
    f.edge((60, y), (140, y), "next"), f.edge((140, y), (205, y), "next")
    f.rect(30, rows["opt"], 220, H, "white", KINDS["neural"][1])
    f.text(140, rows["opt"] + 22, "OPT (operators)", size=13, weight="bold")
    t = rows["opt"] + 45
    f.node(140, t, "−"), f.node(90, t + 42, "^"), f.node(190, t + 42, "y"), f.node(62, t + 82, "x"), f.node(118, t + 82, "2")
    f.edge((140, t), (90, t + 42), "arg1", at=(108, t + 14, "end")), f.edge((140, t), (190, t + 42), "arg2", at=(172, t + 14, "start"))
    f.edge((90, t + 42), (62, t + 82), "arg1", at=(68, t + 62, "end"))
    f.edge((90, t + 42), (118, t + 82), "arg2", at=(112, t + 62, "start"))
    f.text(140, rows["opt"] + H + 18, "example: x² − y", size=11, color=MUTED, italic=True)

    stages = [
        (285, 150, "Node embedding", ["tag + symbol", "(own vectors for", "pad / unknown / none)"]),
        (470, 190, "4 × GATv2 layer", ["edge-type embeddings", "as edge features;", "residual, LayerNorm, GELU"]),
        (695, 165, "Readout", ["attention pooling ‖", "mean pooling", "→ linear"]),
    ]
    for key, top in rows.items():
        prev = 250
        for x, w, title, lines in stages:
            f.box(x, top + 25, w, 100, "neural", title, lines)
            f.path([(prev, top + 75), (x, top + 75)])
            prev = x + w
    fx, fw = 905, 120
    mid = (rows["slt"] + rows["opt"] + H) / 2
    f.box(fx, mid - 55, fw, 110, "neural", "Fusion", ["concatenate", "→ MLP"])
    for top in rows.values():
        f.path([(860, top + 75), (880, top + 75), (880, mid), (fx, mid)])
    f.box(fx - 5, 20, 130, 46, "neural", "missing graph", ["learned vector"], title_size=11.5, line_size=10.5, dashed=True)
    f.path([(fx + fw / 2, 66), (fx + fw / 2, mid - 55)], color=KINDS["neural"][1], dashed=True)
    f.box(1055, mid - 55, 105, 110, "judged", "Formula", ["vector", "256-d,", "L2-normalised"])
    f.path([(fx + fw, mid), (1055, mid)])
    f.text(590, 425, "Hidden size 256, 4 attention heads, dropout 0.1; 12.8M parameters in total.",
           size=11.5, color=MUTED, italic=True)
    return f.save("encoder.svg")


def training() -> Path:
    f = Figure(1180, 470)
    f.panel(20, 20, 560, 430, "(a) Contrastive pretraining · no labels")
    f.panel(600, 20, 560, 430, "(b) Supervised fine-tuning · ARQMath judgments")

    # (a)
    f.box(45, 190, 110, 70, "data", "Formula", ["from the 8.4M", "corpus"])
    f.box(195, 85, 140, 60, "neural", "View 1", ["random augmentations"], line_size=10.5)
    f.box(195, 305, 140, 60, "neural", "View 2", ["random augmentations"], line_size=10.5)
    f.path([(155, 225), (175, 225), (175, 115), (195, 115)])
    f.path([(175, 225), (175, 335), (195, 335)])
    f.rect(185, 165, 160, 120, "white", "#D1D5DB", rx=6, width=1)
    for i, s in enumerate(["consistent variable renaming", "(same in SLT and OPT)", "number swaps",
                           "subtree crop (≤ 40% of nodes)", "drop SLT or OPT (p 0.15)"]):
        f.text(265, 187 + i * 21, s, size=10.5, color=MUTED)
    f.box(380, 85, 90, 60, "neural", "Encoder", ["shared"], line_size=10.5)
    f.box(380, 305, 90, 60, "neural", "Encoder", ["shared"], line_size=10.5)
    f.path([(335, 115), (380, 115)]), f.path([(335, 335), (380, 335)])
    # similarity matrix: rows = view-1 embeddings, columns = view-2 embeddings of the same batch
    mx, my, n, c = 482, 190, 5, 15
    for i in range(n):
        for j in range(n):
            fill = KINDS["neural"][1] if i == j else "#EEE9F8"
            f.rect(mx + j * c, my + i * c, c - 2, c - 2, fill, fill, rx=2, width=0)
    cxm = mx + n * c / 2 - 1
    f.path([(425, 145), (425, 168), (cxm, 168), (cxm, my - 3)])
    f.path([(425, 305), (425, 282), (cxm, 282), (cxm, my + n * c + 2)])
    f.text(300, 400, "InfoNCE over the batch: each view must pick out its partner (the diagonal) among all", size=11, color=MUTED)
    f.text(300, 416, "other formulas in the batch; symmetric loss, learnable temperature τ.", size=11, color=MUTED)
    f.text(300, 438, "Selected model: p(rename) = 0.4; checkpoint chosen by quick dev nDCG′.", size=11, color=MUTED, italic=True)

    # (b)
    inputs = [(70, "judged", "Anchor", ["query formula, or a relevant", "formula (p = 0.5)"]),
              (170, "judged", "Positive", ["formula graded 2–3", "for the same topic"]),
              (270, "judged", "Hard negative", ["formula judged 0–1", "for the same topic"])]
    for y, kind, title, lines in inputs:
        f.box(625, y, 200, 76, kind, title, lines)
        f.path([(825, y + 38), (865, y + 38)])
    f.box(865, 70, 110, 276, "neural", "Encoder", ["initialised", "from the", "pretrained", "model"])
    f.box(1010, 140, 130, 136, "neural", "Loss", ["each anchor vs.", "all positives and", "negatives in", "the batch (2B)"], line_size=10.5)
    f.path([(975, 208), (1010, 208)])
    f.text(880, 385, "B = 64 distinct topics per batch, one example each, so no in-batch false negatives.",
           size=11, color=MUTED, italic=True)
    f.text(880, 405, "600-step schedule, lr 5e-5; development stops at step 250 (chosen by quick dev MAP′).",
           size=11, color=MUTED, italic=True)
    f.text(880, 425, "Development: ARQMath-1 (74 topics). Final model: ARQMath-1 + 2 (144 topics).",
           size=11, color=MUTED, italic=True)
    return f.save("training.svg")


def reranker() -> Path:
    f = Figure(1180, 470)
    f.box(30, 120, 160, 90, "fuse", "Candidates", ["top 300 of the", "RRF run"])
    f.rect(230, 40, 470, 250, "white", "#D1D5DB", rx=10, width=1)
    f.text(465, 64, "48 features per (query, candidate) pair", size=13, weight="bold")
    groups = [
        (80, "Overlap of symbol pairs and 2-edge paths (36)",
         ["recall, precision, F1 × exact / unified / structure labels × SLT, OPT"]),
        (150, "Tree edit similarity, size, missing graph (8)",
         ["Zhang–Shasha TED, exact and unified labels; node-count ratio"]),
        (220, "First-stage scores (4)",
         ["BM25 and pretrained-GNN cosine, raw and min-max normalised"]),
    ]
    for y, title, lines in groups:
        f.box(250, y, 430, 56, "data", title, lines, title_size=12.5, line_size=10.5)
    f.path([(190, 165), (230, 165)])
    f.box(740, 120, 170, 90, "fuse", "Linear score", ["standardise features,", "s = w · x"])
    f.path([(700, 165), (740, 165)])
    f.box(950, 120, 200, 90, "judged", "Reranked list", ["top 300 reordered by s;", "ranks 301–1,000 unchanged"])
    f.path([(910, 165), (950, 165)])

    f.rect(230, 330, 920, 110, PANEL, "#D1D5DB", rx=10, width=1)
    f.text(246, 354, "TRAINING (OFFLINE)", size=11, weight="bold", anchor="start", color=MUTED, spacing=1.2)
    f.box(250, 370, 230, 56, "judged", "Judged pairs", ["same topic, different grades"], line_size=10.5)
    f.box(520, 370, 230, 56, "fuse", "Feature differences", ["x(better) − x(worse)"], line_size=10.5)
    f.box(790, 370, 230, 56, "fuse", "Logistic regression", ["no intercept, C = 0.1"], line_size=10.5)
    f.path([(480, 398), (520, 398)]), f.path([(750, 398), (790, 398)])
    f.path([(905, 370), (905, 300), (825, 300), (825, 210)], color=KINDS["fuse"][1], dashed=True)
    f.label(905, 322, "weights w", color=KINDS["fuse"][1])
    f.text(590, 462, "Depth 300, α = 1 (no interpolation with RRF) and C = 0.1 chosen on ARQMath-2; 5-fold CV estimate within 0.004 nDCG′.",
           size=11, color=MUTED, italic=True)
    return f.save("reranker.svg")


def protocol() -> Path:
    f = Figure(1180, 300)
    phases = [
        (30, "judged", "1 · Train", ["models fit on ARQMath-1", "74 judged topics"]),
        (265, "judged", "2 · Develop", ["all choices made on ARQMath-2", "58 topics; reranker 5-fold CV"]),
        (500, "data", "3 · Freeze", ["encoder, fine-tune step 250,", "RRF k = 60, reranker setting"]),
        (735, "neural", "4 · Retrain", ["same settings on", "ARQMath-1 + 2 (144 topics)"]),
        (970, "fuse", "5 · Test once", ["ARQMath-3, 76 topics", "official cross-check"]),
    ]
    for x, kind, title, lines in phases:
        f.box(x, 70, 190, 90, kind, title, lines)
    for (x, *_), (nx, *_) in zip(phases, phases[1:]):
        f.path([(x + 190, 115), (nx, 115)])
    f.text(590, 40, "Experimental protocol: every setting is fixed before ARQMath-3 is used.", size=13, weight="bold")
    notes = ["Nothing downstream of step 3", "is tuned again; the test run",
             "was evaluated a single time."]
    for i, s in enumerate(notes):
        f.text(1065, 192 + i * 17, s, size=11, color=MUTED, italic=True)
    f.text(30, 210, "Why ARQMath-1 + 2 for the final models: dev has done its job once settings are frozen, and",
           size=11.5, color=MUTED, anchor="start")
    f.text(30, 228, "144 training topics instead of 74 is the standard last step before a single test evaluation.",
           size=11.5, color=MUTED, anchor="start")
    f.text(30, 258, "Collection pretraining (no labels) uses all 8.4M corpus formulas; no ARQMath-3 judgments enter any stage.",
           size=11.5, color=MUTED, anchor="start")
    return f.save("protocol.svg")


if __name__ == "__main__":
    for make in (pipeline, encoder, training, reranker, protocol):
        print("wrote", make())
