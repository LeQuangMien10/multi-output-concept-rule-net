"""
download_cub.py — Tải và giải nén CUB-200-2011 từ data.caltech.edu
==================================================================

Máy chủ lưu trữ của Caltech giới hạn tốc độ theo từng kết nối (đo được
khoảng 0,2–0,3 MB/s mỗi kết nối, trong khi mạng tải file khác ở 60 MB/s), và
hay ngắt kết nối giữa chừng. Script tải song song nhiều đoạn byte (HTTP Range),
mỗi đoạn tự thử lại tới khi đủ đúng số byte, ghi thẳng vào đúng vị trí trong
file đích, rồi giải nén. Với 16 kết nối đo được khoảng 5 MB/s, tức khoảng 4
phút cho file 1,1 GB.

Mỗi request đi qua URL gốc của Caltech, URL này redirect sang một URL lưu trữ
có chữ ký hết hạn sau 60 giây, nên mỗi lần thử lại đều lấy redirect mới.

Usage (Kaggle):
    python -m src.scripts.cub.download_cub --out_dir /kaggle/working/CUB
Kết quả: <out_dir>/CUB_200_2011/... và <out_dir>/attributes.txt
"""
from __future__ import annotations

import argparse
import re
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
EXPECTED_SIZE = 1_150_585_339


def parse_args():
    p = argparse.ArgumentParser(description="Tải song song và giải nén CUB-200-2011.")
    p.add_argument("--out_dir", type=str, default="data/CUB")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk_mb", type=int, default=16)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--keep_archive", action="store_true")
    return p.parse_args()


def fetch_range(start: int, end: int, timeout: int = 300) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "cub-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def total_size() -> int:
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0", "User-Agent": "cub-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        m = re.search(r"/(\d+)$", resp.headers.get("Content-Range", ""))
    if not m:
        raise RuntimeError("server did not return a Content-Range header; range requests unsupported?")
    return int(m.group(1))


def download_chunk(path: Path, start: int, end: int, retries: int) -> int:
    want = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            data = fetch_range(start, end)
            if len(data) == want:
                with open(path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return want
        except Exception:
            pass
        time.sleep(min(2 * attempt, 20))
    raise RuntimeError(f"chunk {start}-{end} failed after {retries} attempts")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "CUB_200_2011.tgz"

    size = total_size()
    if size != EXPECTED_SIZE:
        print(f"[WARN] server reports {size} bytes, expected {EXPECTED_SIZE}")
    with open(archive, "wb") as f:
        f.truncate(size)

    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    print(f"[INFO] {size / 2**20:.0f} MB in {len(ranges)} chunks, {args.workers} workers")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_chunk, archive, s, e, args.retries) for s, e in ranges]
        for fut in as_completed(futures):
            done += fut.result()
            elapsed = time.time() - t0
            print(f"  {done / 2**20:7.0f} / {size / 2**20:.0f} MB  ({done / 2**20 / max(elapsed, 1e-6):.1f} MB/s)",
                  flush=True)

    print("[INFO] extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(out_dir, filter="data")
        except TypeError:   # Python < 3.12 không có tham số filter
            tar.extractall(out_dir)
    if not args.keep_archive:
        archive.unlink()
    n_images = sum(1 for _ in (out_dir / "CUB_200_2011" / "images").rglob("*.jpg"))
    print(f"[DONE] {out_dir / 'CUB_200_2011'}: {n_images} images (expected 11788)")


if __name__ == "__main__":
    main()
