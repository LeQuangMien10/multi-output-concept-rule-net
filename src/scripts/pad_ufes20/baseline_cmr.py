"""
baseline_cmr.py - Baseline CMR (Concept-based Memory Reasoner, Debot et al.,
NeurIPS 2024) cho PAD-UFES-20 (6 concept trieu chung, 6 lop diagnostic).
==============================================================================

Dung lai FitzpatrickCMREncoder/FitzpatrickCMR/_DeviceLoader NGUYEN XI tu
src/scripts/fitzpatrick/baseline_cmr.py (khong hardcode gi rieng cho
Fitzpatrick -- num_concepts/num_labels la tham so) va CMR/InputTypes/
SaveBestModelCallbackVal port trong src/models/baselines/cmr_reasoner.py.

3 diem SUA LAI so voi ban Fitzpatrick (xem review truoc khi trien khai --
KHONG phai copy-paste doi path la xong):

  1. diagnosis_f1 dung f1_macro_nclass (6 lop) thay f1_macro_2class (cung
     ly do voi baseline_neurosymbolic_rules.py).

  2. concept_metrics() nhan them mask [N, num_concepts] va loc THEO TUNG
     CONCEPT khi tinh test concept_acc/concept_f1 -- ban Fitzpatrick khong
     mask gi ca (an toan vi CRL-matched scope luon concept_mask=1 het,
     KHONG con dung o day: PAD-UFES-20 co 'grew'/'changed' chi ~81-88%
     coverage).

  3. UNK concept (~18-19% o 'grew'/'changed') duoc IMPUTE bang gia tri NHI
     PHAN theo LOP DA SO cua concept do tren train (round(base_rate)), thay
     vi coi UNK = absent (hard 0) nhu neu dung nguyen _CMRImageDataset cu.
     Ly do khong sua sau hon: core CMR da port (cmr_reasoner.py) nhan CO
     DINH tuple (x, concepts, y), khong co cho cho mask -- loss BCE tinh
     truc tiep tren toan bo batch_c. Sua dung nghia (them mask that vao
     forward()/training_step()) doi vao phan da port gan nguyen van tu
     paper goc, rui ro lam sai thuat toan CMR that. Impute theo lop da so
     la lua chon da xac nhan voi nguoi dung (ban dau du dinh dung XAC SUAT
     LIEN TUC/base rate, nhung smoke-test that phat hien
     cmr_reasoner.py::training_step goi sklearn.accuracy_score de log
     diagnostic -- ham nay crash "mix of continuous and binary targets"
     neu batch_c lan lon gia tri 0/1 that voi xac suat lien tuc, nen phai
     lam tron impute ve 0/1 -- xem _CMRImageDataset ben duoi). Impute nay
     CHI ap dung cho concepts dua vao
     training_step (train + val loader, vi validation_step cua CMR goi
     lai chinh training_step) -- danh gia concept_acc/concept_f1 tren TEST
     van dung concept THAT + mask THAT (xem eval loop cuoi main()), vi
     forward() o che do eval (khong training, khong intervene) dung
     c_pred cua chinh encoder de tinh y_per_rule, KHONG dung batch_c --
     nen gia tri concepts truyen vao predict_proba() luc eval khong anh
     huong ket qua, chi anh huong ne61u dung de tinh metric truc tiep (da
     tranh bang cach lay concept that/mask that rieng, khong lay tu
     _CMRImageDataset da impute).

CANH BAO RUI RO (ke thua tu ban Fitzpatrick, van dung o day -- 1,606 anh
train, con it hon ca 636 anh CRL-matched): true_log_p_c trong loss cua CMR
la BCE KHONG pos_weight, cung co che gay lazy-collapse da gap o System~1.
FitzpatrickCMR mac dinh dung lai recipe da chung minh tranh duoc collapse do
(backbone_lr_scale=100). Script IN RA train concept accuracy moi epoch de
phat hien som neu van collapse.

Usage (Kaggle):
    !pip install lightning
    python -m src.scripts.pad_ufes20.baseline_cmr \\
        --data_dir /kaggle/working/PAD-UFES-20_prepared \\
        --img_dir /kaggle/input/datasets/lquangmin/pad-ufes-20/images \\
        --seed 42 \\
        --output_dir /kaggle/working/outputs/pad_ufes20_cmr_seed42 \\
        --output_json /kaggle/working/outputs/pad_ufes20_cmr_seed42.json \\
        --output_rules_txt /kaggle/working/outputs/pad_ufes20_cmr_seed42_rules.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.datasets.fitzpatrick.fitzpatrick_dataset import build_transforms
from src.datasets.pad_ufes20.pad_ufes20_dataset import PadUfes20Dataset
from src.models.baselines.cmr_reasoner import InputTypes, SaveBestModelCallbackVal
from src.scripts.fitzpatrick.baseline_cmr import FitzpatrickCMR, FitzpatrickCMREncoder, _DeviceLoader
from src.utils.pad_ufes20_concepts import CONCEPT_NAMES as DEFAULT_CONCEPT_NAMES, LABEL_NAMES as DEFAULT_LABEL_NAMES
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="CMR (Debot et al.) baseline, PAD-UFES-20, full end-to-end.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--label_names", type=str, default=",".join(DEFAULT_LABEL_NAMES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--n_rules", type=int, default=3, help="So rule cho phep moi task.")
    p.add_argument("--rule_emb_size", type=int, default=500)
    p.add_argument("--selector_input", type=str, default="embedding", choices=["embedding", "concepts"])
    p.add_argument("--w_c", type=float, default=1.0)
    p.add_argument("--w_y", type=float, default=30.0)
    p.add_argument("--w_yF", type=float, default=0.005)
    p.add_argument("--reset_selector_every_n_epochs", type=int, default=25)

    p.add_argument("--lr", type=float, default=5e-6, help="LR cho rule module/selector/concept_predictor.")
    p.add_argument("--backbone_lr_scale", type=float, default=100.0,
                    help="LR backbone = lr * backbone_lr_scale. Xem canh bao rui ro dau file.")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=150)

    p.add_argument("--output_dir", type=str, required=True, help="Noi luu checkpoint Lightning tam thoi.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_rules_txt", type=str, default=None,
                    help="Neu dat, ghi rule hoc duoc (aggregate_rules) ra file nay de doc thu cong.")
    return p.parse_args()


# ----- cong thuc metric -----

def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).mean())


def binary_f1(y_true, y_pred, pos_label):
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def f1_macro_2class(y_true, y_pred):
    """F1 macro cho 1 CONCEPT nhi phan (present/absent)."""
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def f1_macro_nclass(y_true, y_pred, num_classes):
    """F1 macro cho NHAN CHAN DOAN -- 6 lop o PAD-UFES-20."""
    f1s = [binary_f1(y_true, y_pred, c) for c in range(num_classes)]
    return sum(f1s) / len(f1s)


def concept_metrics(pred, gt, mask, concept_names):
    """Macro theo CONCEPT, chi tren o BIET (mask[:, i]=1) -- khac ban
    Fitzpatrick (khong mask, an toan vi scope do luon concept_mask=1)."""
    accs, f1s = [], []
    for i in range(len(concept_names)):
        keep_i = mask[:, i].astype(bool)
        true_vars, pred_vars = gt[keep_i, i], pred[keep_i, i]
        if true_vars.sum() == 0:
            continue
        accs.append(accuracy_score(true_vars, pred_vars))
        f1s.append(f1_macro_2class(true_vars, pred_vars))
    return float(sum(accs) / len(accs)), float(sum(f1s) / len(f1s))


def compute_concept_base_rates(train_csv: Path, concept_names: list[str]) -> torch.Tensor:
    """Ty le duong tinh THAT (trong so o biet) cua tung concept tren train --
    dung de impute UNK khi feed vao CMR core (xem docstring dau file). Doc
    thang tu CSV, khong can load anh."""
    with open(train_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rates = []
    for name in concept_names:
        known_vals = [float(r[name]) for r in rows if float(r[f"{name}_mask"]) == 1.0]
        rates.append(sum(known_vals) / len(known_vals) if known_vals else 0.5)
    return torch.tensor(rates, dtype=torch.float32)


class _CMRImageDataset(Dataset):
    """Boc PadUfes20Dataset, tra ve (image, concepts_for_loss, y_onehot).
    Neu impute_values duoc dat, o UNK (mask=0) duoc thay bang ty le duong
    tinh trung binh cua concept do tren train (thay vi coi UNK=absent) --
    xem docstring dau file. Dung cho train/val loader (noi CMR core dung
    truc tiep gia tri nay lam target cho concept BCE loss); KHONG dung cho
    test (eval loop doc thang tu PadUfes20Dataset de lay concept That/mask
    that, xem main())."""

    def __init__(self, base: PadUfes20Dataset, num_labels: int, impute_values: torch.Tensor | None = None):
        self.base = base
        self.num_labels = num_labels
        self.concept_names = base.concept_names
        self.impute_values = impute_values

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, labels = self.base[idx]
        concepts = labels["concepts"]
        if self.impute_values is not None:
            mask = labels["concept_mask"]
            # Lam tron ve 0/1 theo lop da so (khong dung xac suat lien tuc):
            # cmr_reasoner.py (training_step, dong ~391) goi sklearn
            # accuracy_score de log diagnostic (c'_accuracy) tren batch_c --
            # sklearn tu suy ra kieu target (binary/continuous) va SE CRASH
            # ("mix of continuous and binary targets") neu 1 phan tu trong
            # cung mang la xac suat lien tuc con phan con lai la 0/1 that.
            # Phat hien qua smoke-test that (khong doan truoc duoc tu doc
            # code), nen chuyen impute sang gia tri nhi phan theo lop da so
            # cua concept do tren train thay vi giu xac suat lien tuc.
            fill = (self.impute_values > 0.5).float()
            concepts = concepts * mask + fill * (1.0 - mask)
        y_onehot = F.one_hot(labels["label"], num_classes=self.num_labels).float()
        return img, concepts, y_onehot


def make_cmr_loaders(data_dir: Path, img_dir: Path, image_size: int, num_labels: int,
                      batch_size: int, num_workers: int, concept_names: list[str]):
    base_rates = compute_concept_base_rates(data_dir / "train.csv", concept_names)
    print(f"[INFO] Concept base rates (dung de impute UNK khi train/val): "
          f"{dict(zip(concept_names, [round(r, 4) for r in base_rates.tolist()]))}")

    train_base = PadUfes20Dataset(data_dir / "train.csv", img_dir, build_transforms("train", image_size))
    val_base = PadUfes20Dataset(data_dir / "val.csv", img_dir, build_transforms("val", image_size))
    test_base = PadUfes20Dataset(data_dir / "test.csv", img_dir, build_transforms("test", image_size))

    train_ds = _CMRImageDataset(train_base, num_labels, impute_values=base_rates)
    val_ds = _CMRImageDataset(val_base, num_labels, impute_values=base_rates)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    # test_base KHONG boc qua _CMRImageDataset -- eval loop can concept That
    # + mask that (khong impute) de tinh concept_acc/concept_f1 trung thuc.
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def main():
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint

    args = parse_args()
    set_seed(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    label_names = args.label_names.split(",")
    num_labels = len(label_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_cmr_loaders(
        Path(args.data_dir), args.img_dir, args.image_size, num_labels,
        args.train_batch_size, args.num_workers, DEFAULT_CONCEPT_NAMES,
    )
    concept_names = train_loader.dataset.concept_names
    num_concepts = len(concept_names)

    encoder = FitzpatrickCMREncoder(num_concepts, backbone_name=args.backbone,
                                     pretrained=True, dropout=args.dropout)
    selector_input = InputTypes.embedding if args.selector_input == "embedding" else InputTypes.concepts

    model = FitzpatrickCMR(
        encoder=encoder,
        emb_size=encoder.feature_dim,
        rule_emb_size=args.rule_emb_size,
        n_tasks=num_labels,
        n_rules=args.n_rules,
        n_concepts=num_concepts,
        concept_names=concept_names,
        learning_rate=args.lr,
        selector_input=selector_input,
        w_c=args.w_c, w_y=args.w_y, w_yF=args.w_yF,
        reset_selector_every_n_epochs=args.reset_selector_every_n_epochs,
        backbone_lr_scale=args.backbone_lr_scale,
    )

    checkpoint_cb = ModelCheckpoint(dirpath=str(output_dir), save_top_k=1, monitor="val_loss", mode="min")
    best_cb = SaveBestModelCallbackVal()
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[best_cb, checkpoint_cb],
                          accelerator="auto", devices=1, logger=False, enable_progress_bar=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if best_cb.best_state_dict is not None:
        model.load_state_dict(best_cb.best_state_dict)
        print(f"[INFO] Loaded best checkpoint at epoch {best_cb.best_epoch} (val_loss={best_cb.best_loss:.4f})")
    model.to(device)
    model.eval()

    # --- Danh gia tren test: concept That + mask That (KHONG impute), doc
    # truc tiep tu PadUfes20Dataset -- xem docstring dau file ve vi sao gia
    # tri concepts truyen vao predict_proba khong anh huong ket qua o eval
    # mode (forward() dung c_pred cua chinh encoder, khong dung batch_c). ---
    all_c_pred, all_c_gt, all_c_mask, all_probs, all_y_true = [], [], [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            concepts = labels["concepts"].to(device)
            y_onehot = F.one_hot(labels["label"], num_classes=num_labels).float().to(device)
            probs, c_pred = model.predict_proba((images, concepts, y_onehot))
            all_c_pred.append((c_pred > 0.5).float().cpu())
            all_c_gt.append(labels["concepts"])
            all_c_mask.append(labels["concept_mask"])
            all_probs.append(probs.cpu())
            all_y_true.append(labels["label"])

    c_pred = torch.cat(all_c_pred).numpy().astype(int)
    c_gt = torch.cat(all_c_gt).numpy().astype(int)
    c_mask = torch.cat(all_c_mask).numpy()
    concept_acc, concept_f1 = concept_metrics(c_pred, c_gt, c_mask, concept_names)

    y_prob = torch.cat(all_probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    y_true = torch.cat(all_y_true).numpy()
    diagnosis_acc = accuracy_score(y_true, y_pred)
    diagnosis_f1 = f1_macro_nclass(y_true, y_pred, num_labels)

    result = {
        "seed": args.seed,
        "n_concept_eval": int(c_mask.sum()),
        "concept_acc": concept_acc,
        "concept_f1": concept_f1,
        "diagnosis_acc": diagnosis_acc,
        "diagnosis_f1": diagnosis_f1,
        "n_rules": args.n_rules,
        "rule_emb_size": args.rule_emb_size,
        "selector_input": args.selector_input,
        "lr": args.lr,
        "backbone_lr_scale": args.backbone_lr_scale,
        "best_val_loss": float(best_cb.best_loss) if best_cb.best_state_dict is not None else None,
        "best_epoch": best_cb.best_epoch,
    }
    print(json.dumps(result, indent=2))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    if args.output_rules_txt:
        with torch.no_grad():
            rule_vars = model.get_all_rule_vars()  # [n_tasks, n_rules, n_concepts, 3] pos/neg/irr
            try:
                _, task_to_rule_idx = model.aggregate_rules(_DeviceLoader(train_loader, device), type="most_likely")
            except Exception as e:
                print(f"[WARN] aggregate_rules (tan suat rule duoc chon) loi, bo qua: {e}")
                task_to_rule_idx = None

        lines = []
        for task in range(num_labels):
            used_idx = task_to_rule_idx[task] if task_to_rule_idx is not None else None
            lines.append(f"=== Task {label_names[task]} ==="
                         + (f"  (rule duoc chon tren train: {sorted(used_idx)})" if used_idx else ""))
            for rule_idx in range(args.n_rules):
                c_type = torch.argmax(rule_vars[task, rule_idx], dim=-1)  # 0=pos,1=neg,2=irr moi concept
                pos = [concept_names[k] for k in range(len(c_type)) if c_type[k] == 0]
                neg = [concept_names[k] for k in range(len(c_type)) if c_type[k] == 1]
                n_irr = num_concepts - len(pos) - len(neg)
                lines.append(f"  rule {rule_idx}: {len(pos)} required-present, {len(neg)} required-absent, {n_irr} irrelevant")
                if pos:
                    lines.append(f"    required present: {', '.join(pos)}")
                if neg:
                    lines.append(f"    required absent:  {', '.join(neg)}")
        Path(args.output_rules_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_rules_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
