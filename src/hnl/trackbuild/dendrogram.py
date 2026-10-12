"""Single-linkage dendrogram of a scored graph and its learned, locally adaptive cut (LDC; Stage 1 C, learned track
building).

Connected components at a score cut s are a horizontal cut of the single-linkage dendrogram of the edge scores: Kruskal
on the edges in descending score creates one internal node per merge, at the merging edge's score (its birth level);
a node exists as a component for cuts s with birth > s >= death, where death is its parent's birth level (0 for a
root). Instead of one global s, LDC selects a set of disjoint nodes maximising sum_v (w(v)) with w(v) = q(v) - theta,
where q(v) is a learned probability that node v is a correct track (hnl.trackbuild.ldc uses
w(v) = (q(v) - theta) * size(v): hits placed in correct tracks, so of two nested correct nodes the larger wins); the
exact bottom-up recursion best(v) = max(w(v), best(left) + best(right)) solves it in O(n). CC at any single cut is a feasible solution
(w(v) = 1 for the nodes alive and maximal at s), so LDC strictly generalises the preregistered builder; theta is its only
working-point parameter (chosen with the builder rule on val). Labels follow hnl.trackbuild.builders (-1 = no track).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    n: int                   # leaves (hits) 0..n-1; internal nodes n..n+m-1 in creation order
    left: np.ndarray         # (m,) child node ids
    right: np.ndarray
    level: np.ndarray        # (m,) birth level = score of the merging edge (descending in creation order)
    parent: np.ndarray       # (n+m,) parent node id, -1 for roots (leaves included)
    size: np.ndarray         # (n+m,) number of hits
    order: np.ndarray        # (n,) leaves in DFS order: node v's hits are order[start[v]:stop[v]]
    start: np.ndarray        # (n+m,)
    stop: np.ndarray

    @property
    def m(self) -> int:
        return len(self.level)

    def death(self) -> np.ndarray:
        """(m,) death level of every internal node: its parent's birth level, 0 for roots."""
        p = self.parent[self.n:]
        return np.where(p >= 0, self.level[np.maximum(p - self.n, 0)], 0.0)

    def hits(self, v: int) -> np.ndarray:
        return self.order[self.start[v]:self.stop[v]]


def single_linkage(n: int, a: np.ndarray, b: np.ndarray, score: np.ndarray) -> Tree:
    """Kruskal on edges in descending score (ties: input order); every union of two components is a new node."""
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    uf = np.arange(n)
    comp = np.arange(n)                      # tree node of the component whose union-find root is the index
    left, right, level = [], [], []

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    for e in order:
        ra, rb = find(a[e]), find(b[e])
        if ra == rb:
            continue
        k = n + len(level)
        left.append(comp[ra])
        right.append(comp[rb])
        level.append(score[e])
        uf[rb] = ra
        comp[ra] = k
    m = len(level)
    left, right, level = np.array(left, dtype=np.int64), np.array(right, dtype=np.int64), np.array(level)
    parent = np.full(n + m, -1, dtype=np.int64)
    size = np.ones(n + m, dtype=np.int64)
    for k in range(m):
        parent[left[k]] = parent[right[k]] = n + k
        size[n + k] = size[left[k]] + size[right[k]]
    # leaf order by iterative DFS from every root (roots in increasing id), contiguous slices per node
    start = np.zeros(n + m, dtype=np.int64)
    stop = np.zeros(n + m, dtype=np.int64)
    leaf_order = np.empty(n, dtype=np.int64)
    pos = 0
    for root in np.flatnonzero(parent < 0):
        stack = [(int(root), False)]
        while stack:
            v, done = stack.pop()
            if v < n:
                start[v], stop[v] = pos, pos + 1
                leaf_order[pos] = v
                pos += 1
                continue
            if done:
                stop[v] = stop[right[v - n]]
                continue
            start[v] = pos
            stack.append((v, True))
            stack.append((int(right[v - n]), False))
            stack.append((int(left[v - n]), False))
    return Tree(n, left, right, level, parent, size, leaf_order, start, stop)


def dp_cut(tree: Tree, w: np.ndarray) -> np.ndarray:
    """Labels of the disjoint internal nodes maximising sum of w (w: (m,) per internal node; leaves count 0 and are
    never tracks). A node is taken if its weight is positive and at least the best of its subtrees (ties: the larger
    node)."""
    n, m = tree.n, tree.m
    best = np.zeros(n + m)
    take = np.zeros(m, dtype=bool)
    for k in range(m):                       # children are created before parents
        sub = best[tree.left[k]] + best[tree.right[k]]
        if w[k] > 0 and w[k] >= sub:
            best[n + k], take[k] = w[k], True
        else:
            best[n + k] = sub
    labels = np.full(n, -1, dtype=np.int64)
    t = 0
    stack = [int(r) for r in np.flatnonzero(tree.parent < 0) if r >= n]
    while stack:
        v = stack.pop()
        k = v - n
        if take[k]:
            labels[tree.hits(v)] = t
            t += 1
        else:
            for c in (tree.left[k], tree.right[k]):
                if c >= n:
                    stack.append(int(c))
    return labels


def cc_weights(tree: Tree, cut: float) -> np.ndarray:
    """w(v) = 1 for the nodes alive and maximal at cut s (birth > s >= death): dp_cut then equals CC at s."""
    return ((tree.level > cut) & (tree.death() <= cut)).astype(np.float64)
