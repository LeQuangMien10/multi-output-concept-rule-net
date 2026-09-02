"""
baseline_cmr.py - Baseline CMR (Concept-based Memory Reasoner, Debot et al.,
NeurIPS 2024) cho Fitzpatrick17k, scope CRL-matched (48 concept SkinCon,
2 lop benign/malignant). Dung class da port trong
src/models/baselines/cmr_reasoner.py (xem docstring file do de biet chinh
xac phan nao giu nguyen/bo tu ban goc).

Khac Basci et al. (trich concept post-hoc tu 1 classifier CHI train bang
nhan): CMR gia dinh mot concept encoder DA CO SAN (kieu CBM) roi hoc mot
"neural rule selector" chon 1 trong N rule logic kha hoc tu 1 memory
tuong minh cho tung task, danh gia rule do bang symbolic evaluation --
gan voi chinh System~2 cua ICRL (rule memory + match) hon la Basci.

CHAY HOAN TOAN END-TO-END, KHONG DUNG LAI checkpoint System~1 cua ICRL
(quyet dinh cua nguoi dung, doi tu ban truoc): `FitzpatrickCMREncoder` la
1 ResNet-50 pretrained ImageNet rieng (kien truc mirror MNISTEncoder cua
chinh ho -- backbone -> concept_predictor + embedding cho selector, nhung
dung ResNet-50 thay CNN nho vi anh da lieu that, giong triet ly System~1),
train JOINT cung rule module/selector cua CMR trong 1 vong Lightning duy
nhat, tu dau den cuoi qua images.

CANH BAO RUI RO (quan trong, doc truoc khi chay): thanh phan
`true_log_p_c` trong loss cua CMR (concept prediction accuracy cua chinh
encoder) la BCE KHONG pos_weight -- CUNG CO CHE da gay "lazy collapse"
(du doan gan nhu am tinh cho moi concept) khi train System~1 tren dung
scope CRL-matched (636 anh train, concept mat can bang nang) trong qua
trinh dieu tra o phan Table 1 (xem
outputs/fitzpatrick_system1_crlmatched_v2 vs v3, memory
project-crlmatched-recipe-unification). Vi CMR dung CHUNG so anh train va
CHUNG co che BCE khong trong so nay, `FitzpatrickCMR` (subclass ben duoi)
mac dinh dung LAI cong thuc LR da chung minh thuc nghiem tranh duoc
collapse do cho System~1 (`backbone_lr_scale=100` -- backbone train
NHANH HON head, nguoc truc giac, xem
outputs/fitzpatrick_system1_crlmatched_v3/metrics.json). Day la SUY DOAN
CO CO SO (cung dataset, cung co che mat can bang), KHONG PHAI da kiem
chung rieng cho loss phuc tap hon cua CMR (co them w_c/w_y/w_yF khac
trong so) -- vi vay script IN RA train concept accuracy MOI EPOCH de phat
hien som neu collapse van xay ra du da ap dung recipe nay; neu thay
concept accuracy dung o gan ty le da so (khong tang), dieu chinh --lr/
--backbone_lr_scale/--max_epochs truoc khi tin ket qua.

Da smoke-test toan bo pipeline (CMR.forward/training_step/validation_step/
predict_proba/aggregate_rules) cuc bo bang du lieu gia + lightning 2.6.5
that (khong phai mock) -- pass, nhung KHONG the smoke-test rui ro collapse
o tren cuc bo (can du lieu that + nhieu epoch). Rui ro con lai la (1) moi
truong GPU/lightning cua Kaggle, (2) collapse nhu mo ta o tren.

Usage (Kaggle):
    !pip install lightning
    python -m src.scripts.fitzpatrick.baseline_cmr \\
        --data_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k-crl-matched \\
        --img_dir /kaggle/input/datasets/lquangmin/fitzpatrick17k/data/finalfitz17k \\
        --seed 42 \\
        --output_dir /kaggle/working/outputs/cmr_seed42 \\
        --output_json /kaggle/working/outputs/cmr_seed42.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset

from src.datasets.fitzpatrick.fitzpatrick_dataset import FitzpatrickDataset, build_transforms
from src.models.baselines.cmr_reasoner import CMR, InputTypes, SaveBestModelCallbackVal
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="CMR (Debot et al.) baseline, CRL-matched scope, full end-to-end.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--label_names", type=str, default="benign,malignant")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--dropout", type=float, default=0.3)

    # Hyperparam CMR -- mac dinh theo recipe CUB cua chinh ho (dataset gan nhat
    # voi Fitzpatrick17k: anh + concept nhi phan + phan lop).
    p.add_argument("--n_rules", type=int, default=3, help="So rule cho phep moi task.")
    p.add_argument("--rule_emb_size", type=int, default=500)
    p.add_argument("--selector_input", type=str, default="embedding", choices=["embedding", "concepts"])
    p.add_argument("--w_c", type=float, default=1.0)
    p.add_argument("--w_y", type=float, default=30.0)
    p.add_argument("--w_yF", type=float, default=0.005)
    p.add_argument("--reset_selector_every_n_epochs", type=int, default=25)

    # LR/epoch -- mac dinh theo recipe da tranh duoc lazy-collapse o System~1
    # scope CRL-matched (xem canh bao rui ro dau file), KHONG phai gia tri goc
    # cua ho (thiet ke cho encoder MLP nho tren feature co san, khong phai
    # ResNet-50 pretrained).
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


# ----- cong thuc metric giong het eval_crlmatched_metrics.py / baseline_neurosymbolic_rules.py -----

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
    return (binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2


def concept_metrics(pred, gt, concept_names):
    """Cong thuc CRL that: macro theo CONCEPT, bo qua concept GT toan 0."""
    accs, f1s = [], []
    for i in range(len(concept_names)):
        true_vars, pred_vars = gt[:, i], pred[:, i]
        if true_vars.sum() == 0:
            continue
        accs.append(accuracy_score(true_vars, pred_vars))
        f1s.append(f1_macro_2class(true_vars, pred_vars))
    return float(sum(accs) / len(accs)), float(sum(f1s) / len(f1s))


class FitzpatrickCMREncoder(nn.Module):
    """CNN encoder RIENG cho CMR, KHONG dung lai checkpoint System~1 cua
    ICRL. Kien truc mirror MNISTEncoder cua chinh ho (backbone -> mot
    nhanh concept_predictor tra ve c_probs, dac trung backbone tho dung
    lam embedding cho selector) nhung dung ResNet-50 pretrained ImageNet
    thay CNN nho -- cung ly do voi FitzpatrickSystem1 cua ICRL: anh da
    lieu that, chi ~636 anh train o scope CRL-matched, qua nho de hoc dac
    trung thi giac tu dau."""

    FEATURE_DIMS = {"resnet50": 2048, "resnet18": 512}

    def __init__(self, num_concepts: int, backbone_name: str = "resnet50",
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
        elif backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = self.FEATURE_DIMS[backbone_name]
        self.dropout = nn.Dropout(dropout)
        self.concept_predictor = nn.Linear(self.feature_dim, num_concepts)

    def forward(self, x):
        feats = self.backbone(x)
        c_probs = torch.sigmoid(self.concept_predictor(self.dropout(feats)))
        return c_probs, feats


class FitzpatrickCMR(CMR):
    """Subclass CHI de doi configure_optimizers -- ban goc dung 1 LR chung
    cho toan bo tham so, hop ly khi encoder la MLP nho tren feature co
    san (CUB) nhung se pha vo pretrained ResNet-50 neu ap dung nguyen
    LR danh cho rule module/selector moi khoi tao. Khong sua bat ky logic
    thuat toan nao khac cua CMR/MNISTModel."""

    def __init__(self, *args, backbone_lr_scale: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_lr_scale = backbone_lr_scale

    def configure_optimizers(self):
        backbone_ids = {id(p) for p in self.encoder.backbone.parameters()}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return torch.optim.AdamW([
            {"params": list(self.encoder.backbone.parameters()), "lr": self.lr * self.backbone_lr_scale},
            {"params": other_params, "lr": self.lr},
        ])

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if len(self.info["c_accuracy"]) > 0:
            c_acc = sum(self.info["c_accuracy"]) / len(self.info["c_accuracy"])
            print(f"  [monitor] train concept acc = {c_acc:.4f}  (theo doi lazy-collapse, xem canh bao dau file)")


class _CMRImageDataset(Dataset):
    """Boc FitzpatrickDataset de tra ve dung tuple (image, concepts, y_onehot)
    ma CMR.forward mong doi -- CRL-matched scope luon co concept_mask=1
    (xem prepare_dataset_crl_matched.py), nen khong can loc mask o day."""

    def __init__(self, base: FitzpatrickDataset, num_labels: int):
        self.base = base
        self.num_labels = num_labels
        self.concept_names = base.concept_names

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, labels = self.base[idx]
        y_onehot = F.one_hot(labels["label"], num_classes=self.num_labels).float()
        return img, labels["concepts"], y_onehot


def make_cmr_loaders(data_dir: Path, img_dir: Path, image_size: int, num_labels: int,
                      batch_size: int, num_workers: int):
    train_ds = _CMRImageDataset(FitzpatrickDataset(data_dir / "train.csv", img_dir, build_transforms("train", image_size)), num_labels)
    val_ds = _CMRImageDataset(FitzpatrickDataset(data_dir / "val.csv", img_dir, build_transforms("val", image_size)), num_labels)
    test_ds = _CMRImageDataset(FitzpatrickDataset(data_dir / "test.csv", img_dir, build_transforms("test", image_size)), num_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
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
        args.train_batch_size, args.num_workers,
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
    # devices=1 co dinh: Kaggle GPU x2 mac dinh se bat DDP neu de "auto", lam
    # trainer.model.state_dict() (dung trong SaveBestModelCallbackVal) co tien
    # to "module." va lam model.load_state_dict() ben duoi bao loi key-mismatch
    # ngay truoc buoc ghi JSON -- da xac nhan qua 1 lan chay that tren Kaggle.
    # 636 anh train qua nho de can toi 2 GPU, khong danh doi rui ro nay lay toc do.
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[best_cb, checkpoint_cb],
                          accelerator="auto", devices=1, logger=False, enable_progress_bar=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if best_cb.best_state_dict is not None:
        model.load_state_dict(best_cb.best_state_dict)
        print(f"[INFO] Loaded best checkpoint at epoch {best_cb.best_epoch} (val_loss={best_cb.best_loss:.4f})")
    model.to(device)
    model.eval()

    # --- Danh gia tren test: 1 pass qua anh that, encoder cua chinh CMR (khong con System~1) ---
    all_c_pred, all_c_gt, all_probs, all_y_true = [], [], [], []
    with torch.no_grad():
        for images, concepts, y_onehot in test_loader:
            images, concepts, y_onehot = images.to(device), concepts.to(device), y_onehot.to(device)
            probs, c_pred = model.predict_proba((images, concepts, y_onehot))
            all_c_pred.append((c_pred > 0.5).float().cpu())
            all_c_gt.append(concepts.cpu())
            all_probs.append(probs.cpu())
            all_y_true.append(y_onehot.argmax(dim=1).cpu())

    c_pred = torch.cat(all_c_pred).numpy().astype(int)
    c_gt = torch.cat(all_c_gt).numpy().astype(int)
    concept_acc, concept_f1 = concept_metrics(c_pred, c_gt, concept_names)

    y_prob = torch.cat(all_probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    y_true = torch.cat(all_y_true).numpy()
    diagnosis_acc = accuracy_score(y_true, y_pred)
    diagnosis_f1 = f1_macro_2class(y_true, y_pred)

    result = {
        "seed": args.seed,
        "n_concept_eval": int(c_gt.shape[0]),
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
            task_to_rules, _ = model.aggregate_rules(train_loader, type="most_likely")
        lines = []
        for task in range(num_labels):
            lines.append(f"=== Task {label_names[task]} = True ===")
            for rule, support in task_to_rules[task].items():
                lines.append(f"  {rule}  (support={support})")
        Path(args.output_rules_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_rules_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
