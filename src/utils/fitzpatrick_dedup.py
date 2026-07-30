"""
fitzpatrick_dedup.py — Perceptual near-duplicate detection dùng chung
=========================================================================

Tách ra từ src/scripts/fitzpatrick/analyze_metadata.py để prepare_dataset.py
(chia train/val/test) và analyze_metadata.py (báo cáo rủi ro) dùng chung đúng
1 định nghĩa near-dup, không lệch nhau giữa 2 script.

dHash 16x16 (256-bit, so sánh gradient ngang giữa pixel liền kề) — KHÔNG dùng
average-hash (aHash) vì đã kiểm chứng aHash 8x8 match theo tông màu da/độ sáng
tổng thể chứ không theo cấu trúc tổn thương, gây false positive nặng (~31% ảnh
"gần trùng" giả, nhiều cặp có nhãn chẩn đoán khác hẳn nhau). dHash phân biệt
tốt hơn nhiều: cùng ngưỡng kiểm tra, ~93% cặp near-dup phát hiện được có cùng
nhãn chẩn đoán — đúng bản chất "ảnh chụp lại/crop nhẹ của cùng 1 tổn thương".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DHASH_SIZE = 16          # 16x16 -> 256-bit difference hash
DUP_HAMMING_STRICT = 10   # <=10/256 bit khác nhau (~4%) -> gần như chắc chắn trùng
DUP_HAMMING_LOOSE = 25    # <=25/256 bit khác nhau (~10%) -> đáng xem lại thủ công

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compute_dhashes(img_dir: Path, image_files: list[str]) -> np.ndarray:
    """16x16 difference-hash -> [n,4] uint64 (256 bit/ảnh)."""
    n = len(image_files)
    hashes = np.zeros((n, 4), dtype=np.uint64)
    for i, fn in enumerate(image_files):
        img = Image.open(img_dir / fn).convert("L").resize((DHASH_SIZE + 1, DHASH_SIZE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        diff = (arr[:, :-1] > arr[:, 1:]).flatten()   # 256 bool
        hashes[i] = np.packbits(diff).view(np.uint64)
    return hashes


def hamming_popcount256(x: np.ndarray) -> np.ndarray:
    """x: [..., 4] uint64 (256-bit hash) -> popcount 0..256."""
    b = x.view(np.uint8).reshape(*x.shape[:-1], 4, 8)
    return _POPCOUNT_TABLE[b].sum(axis=(-1, -2))


def find_near_duplicates(hashes: np.ndarray, chunk: int = 250) -> tuple[list, list]:
    """Trả về (strict_pairs, loose_pairs), mỗi phần tử (i, j, hamming_dist), i<j."""
    n = len(hashes)
    strict_pairs, loose_pairs = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = hashes[start:end][:, None, :]     # [c,1,4]
        xor = block ^ hashes[None, :, :]           # [c,n,4]
        dist = hamming_popcount256(xor)             # [c,n]
        for local_i in range(end - start):
            i = start + local_i
            row = dist[local_i, i + 1:]
            for j in np.nonzero(row <= DUP_HAMMING_STRICT)[0] + i + 1:
                strict_pairs.append((i, int(j), int(dist[local_i, j])))
            for j in np.nonzero((row > DUP_HAMMING_STRICT) & (row <= DUP_HAMMING_LOOSE))[0] + i + 1:
                loose_pairs.append((i, int(j), int(dist[local_i, j])))
    return strict_pairs, loose_pairs


class UnionFind:
    """Union-find tối giản — gom ảnh near-dup (strict+loose) thành 1 nhóm để
    chia split theo nhóm, tránh 1 lesion vừa có ảnh ở train vừa có ở test."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out
