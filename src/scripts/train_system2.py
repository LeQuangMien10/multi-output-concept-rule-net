import argparse
import torch
from torch.utils.data import DataLoader

from src.datasets.mnist_math_dataset import MNISTMathPTDataset
from src.models.concept_extractor import ConceptExtractor
from src.models.rule_memory import RuleMemory
from src.models.system2_model import System2


def parse_args():
    return argparse.ArgumentParser().parse_args()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------
    # LOAD DATA
    # -------------------
    dataset = MNISTMathPTDataset("/kaggle/input/mnist-math/train.pt")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # -------------------
    # SYSTEM 1 (frozen)
    # -------------------
    system1_ckpt = "/kaggle/working/outputs/system1_baseline/best_model.pt"
    extractor = ConceptExtractor(system1_ckpt, device=device)

    # -------------------
    # SYSTEM 2
    # -------------------
    rule_memory = RuleMemory(num_rules=64).to(device)
    system2 = System2(rule_memory).to(device)

    system2.eval()

    correct = 0
    total = 0

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

    print("SYSTEM2 ACC:", correct / total)


if __name__ == "__main__":
    main()