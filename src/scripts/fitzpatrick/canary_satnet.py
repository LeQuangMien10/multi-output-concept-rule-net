"""
canary_satnet.py - Kiem tra nhanh SATNet (locuslab/SATNet, ICML 2019) co
build/chay duoc tren moi truong hien tai khong, TRUOC khi thiet ke pipeline
day du cho Fitzpatrick17k.

Ly do can canary test rieng: SATNet la CUDA extension da lau khong duoc
bao tri, co issue da biet, chua fix ve loi build tren CUDA 12.x/PyTorch
2.2+ (satnet_cuda.cu: identifier "saturate" is undefined -- xem
github.com/locuslab/SATNet/issues/17), cong voi 1 issue cu bao ca duong
CPU cung co van de. Kaggle GPU hien tai gan nhu chac chan dung CUDA 12.x,
nen rui ro build fail la that, khong phai gia thuyet. Chay canary nay
truoc de biet ngay co nen tiep tuc huong SATNet hay chuyen sang CMR
(Debot et al., NeurIPS -- xem ke hoach) neu build fail, tranh thiet ke
uong phi.

Usage (Kaggle, GPU notebook):
    !pip install satnet
    !python -m src.scripts.fitzpatrick.canary_satnet
"""
from __future__ import annotations

import sys

import torch


def main():
    print(f"[INFO] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}"
          f"  cuda_version={torch.version.cuda}")

    try:
        import satnet
    except Exception as e:
        print(f"[FAIL] import satnet: {type(e).__name__}: {e}")
        print("\n[RESULT] SATNet KHONG import duoc tren moi truong nay -- "
              "xem xet chuyen sang CMR (Debot et al.) thay the.")
        sys.exit(1)
    print(f"[OK] import satnet thanh cong (satnet.__file__={satnet.__file__})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, m, aux, batch = 10, 4, 2, 8

    try:
        layer = satnet.SATNet(n, m, aux).to(device)
        z = torch.rand(batch, n, device=device)
        is_input = torch.zeros(batch, n, dtype=torch.int32, device=device)
        is_input[:, :6] = 1  # 6 bit "known" (gia lap concept), 4 bit can du doan (gia lap nhan)

        out = layer(z, is_input)
        print(f"[OK] forward thanh cong, out.shape={tuple(out.shape)}")

        loss = out.sum()
        loss.backward()
        print("[OK] backward thanh cong (co gradient qua SATNet layer)")

        print(f"\n[RESULT] SATNet build va chay OK tren moi truong nay (device={device}). "
              "An toan de thiet ke pipeline day du cho Fitzpatrick17k.")
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        print("\n[RESULT] SATNet KHONG chay duoc tren moi truong nay -- "
              "xem xet chuyen sang CMR (Debot et al.) thay the.")
        sys.exit(1)


if __name__ == "__main__":
    main()
