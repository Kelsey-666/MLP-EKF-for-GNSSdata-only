"""Tiny linear algebra helpers for small dense matrices."""

from __future__ import annotations

from typing import List, Optional


def eye(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    k = len(b)
    m = len(b[0])
    out = zeros(n, m)
    for i in range(n):
        for t in range(k):
            ai = a[i][t]
            if ai == 0.0:
                continue
            bt = b[t]
            for j in range(m):
                out[i][j] += ai * bt[j]
    return out


def mat_transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]


def solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]