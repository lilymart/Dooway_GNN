import os
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

results_dir = args.results_dir
seed = args.seed


# ------------------------------------------------------------
# Load probabilities
# ------------------------------------------------------------

text_path = os.path.join(
    results_dir,
    f"probs_seed{seed}_attrsTrue_imgsFalse.pt"
)

structure_path = os.path.join(
    results_dir,
    f"probs_seed{seed}_attrsFalse_imgsFalse.pt"
)

text_data = torch.load(text_path)
structure_data = torch.load(structure_path)

p_text = text_data["prob"]
p_struct = structure_data["prob"]

idx_text = text_data["test_idx"]
idx_struct = structure_data["test_idx"]


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

assert p_text.shape == p_struct.shape, (
    f"Different probability shapes: "
    f"{p_text.shape} vs {p_struct.shape}"
)

assert torch.equal(idx_text, idx_struct), (
    "The two experiments do not use exactly the same test nodes."
)

print(f"Seed: {seed}")
print(f"Probability tensor shape: {p_text.shape}")
print(f"Number of test nodes: {len(idx_text)}")


# ------------------------------------------------------------
# Probability differences
# ------------------------------------------------------------

diff = (p_text - p_struct).abs()

print("\n=== PROBABILITY DIFFERENCES ===")
print(f"Mean absolute difference   : {diff.mean().item():.8f}")
print(f"Median absolute difference : {diff.median().item():.8f}")
print(f"Max absolute difference    : {diff.max().item():.8f}")


# ------------------------------------------------------------
# Binary predictions at threshold 0.5
# ------------------------------------------------------------

pred_text = p_text > 0.5
pred_struct = p_struct > 0.5

different = pred_text != pred_struct

n_different = different.sum().item()
n_total = different.numel()

print("\n=== BINARY PREDICTION DIFFERENCES ===")
print(f"Different decisions: {n_different}/{n_total}")
print(
    f"Fraction of different decisions: "
    f"{n_different / n_total:.8f}"
)


# ------------------------------------------------------------
# Per-class analysis
# ------------------------------------------------------------

print("\n=== PER-CLASS ANALYSIS ===")

for c in range(p_text.shape[1]):

    class_diff = (p_text[:, c] - p_struct[:, c]).abs()

    class_pred_diff = (
        pred_text[:, c] != pred_struct[:, c]
    )

    print(
        f"Class {c:2d} | "
        f"mean abs diff = {class_diff.mean().item():.8f} | "
        f"max abs diff = {class_diff.max().item():.8f} | "
        f"changed decisions = "
        f"{class_pred_diff.sum().item():4d}/{len(class_pred_diff)}"
    )