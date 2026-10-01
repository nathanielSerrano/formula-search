"""
Tree edit distance (Zhang & Shasha, 1989) with unit costs.

The minimum number of node insertions, deletions and relabellings that turn one
ordered tree into another. The organizers' TangentCFT2ED run reranks Tangent-CFT
results with tree edit distance; here it is one of the reranker's features.

Trees are given as (labels, children) with node 0 as the root and children listed in
order. Runtime is O(n·m·min(depth, leaves)²), fine for formula trees of a few dozen
nodes.
"""

from __future__ import annotations

from typing import Hashable, List, Sequence, Tuple

Tree = Tuple[Sequence[Hashable], Sequence[Sequence[int]]]  # (labels, children)


def _postorder(tree: Tree) -> Tuple[List[Hashable], List[int], List[int]]:
    """Labels in postorder, leftmost-leaf index of each node, and the keyroots."""
    labels, children = tree
    order: List[int] = []
    stack = [(0, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        else:
            stack.append((node, True))
            stack.extend((c, False) for c in reversed(children[node]))
    position = {node: i for i, node in enumerate(order)}
    lml = [0] * len(order)
    for node in order:  # children come before their parent in postorder
        kids = children[node]
        lml[position[node]] = lml[position[kids[0]]] if kids else position[node]
    keyroots, seen = [], set()
    for i in range(len(order) - 1, -1, -1):  # highest node for each distinct leftmost leaf
        if lml[i] not in seen:
            seen.add(lml[i])
            keyroots.append(i)
    return [labels[n] for n in order], lml, sorted(keyroots)


def tree_edit_distance(a: Tree, b: Tree) -> int:
    la, lml_a, keys_a = _postorder(a)
    lb, lml_b, keys_b = _postorder(b)
    n, m = len(la), len(lb)
    if n == 0 or m == 0:
        return n + m
    td = [[0] * m for _ in range(n)]
    for i in keys_a:
        for j in keys_b:
            ai, bj = lml_a[i], lml_b[j]
            rows, cols = i - ai + 2, j - bj + 2
            fd = [[0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, rows):
                for y in range(1, cols):
                    ix, jy = ai + x - 1, bj + y - 1
                    if lml_a[ix] == ai and lml_b[jy] == bj:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[x - 1][y - 1] + (la[ix] != lb[jy]))
                        td[ix][jy] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[lml_a[ix] - ai][lml_b[jy] - bj] + td[ix][jy])
    return td[n - 1][m - 1]
