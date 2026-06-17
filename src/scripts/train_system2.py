import argparse
import torch
from torch.utils.data import DataLoader

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.concept_extractor import ConceptExtractor
from src.models.rule_memory import RuleMemory
from src.models.system2_model import System2


def parse_args():
    parser = argparse.ArgumentParser()

    # -------------------
    # DATA
    # -------------------
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset file (train.pt)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )

    # -------------------
    # SYSTEM 1 CKPT
    # -------------------
    parser.add_argument(
        "--system1_ckpt",
        type=str,
        required=True,
        help="Path to System1 checkpoint (best_model.pt)",
    )

    # -------------------
    # SYSTEM 2
    # -------------------
    parser.add_argument(
        "--num_rules",
        type=int,
        default=64,
        help="Number of rules in rule memory",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (auto if None)",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    # -------------------
    # LOAD DATA
    # -------------------
    dataset = MNISTMathPTDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -------------------
    # SYSTEM 1 (frozen)
    # -------------------
    print(f"[INFO] Loading System1 from: {args.system1_ckpt}")
    extractor = ConceptExtractor(args.system1_ckpt, device=device)

    # -------------------
    # SYSTEM 2
    # -------------------
    rule_memory = RuleMemory(num_rules=args.num_rules).to(device)
    system2 = System2(rule_memory).to(device)

    system2.eval()

    correct = 0
    total = 0

    # -------------------
    # EVAL LOOP
    # -------------------
    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            # 1. System1 → concepts
            z = extractor.predict_concepts(images)

            # 2. System2 → reasoning
            logits, best_rule, weight = system2.match(z)

            pred = logits.argmax(dim=-1)

            y = labels["valid"].to(device)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total

    print("\n====================")
    print("SYSTEM 2 RESULT")
    print("====================")
    print(f"Accuracy: {acc:.4f}")
    print(f"Total samples: {total}")
    print(f"10 best rules: {best_rule[:10]}")


if __name__ == "__main__":
    main()