from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from evaluate_one_epoch import evaluate_one_epoch
from CPEN import CPEN
import json
import numpy as np
import torch
import h5py
import os
from scipy.io import loadmat
from glob import glob

from torch.utils.data import DataLoader, random_split
from Dataset import FirstDataset, collate_batch

h5_path = '/home/ltopalis/Desktop/image-alignment-using-nn/datasets/myYaleCroppedA/test/dataset.hdf5'
model_save_path = "/home/ltopalis/Desktop/image-alignment-using-nn/pretrained_models/"
fig_save_path = "pretrained_models/images/testing_with_myYaleCroppedA"
batch = 20

os.makedirs(fig_save_path, exist_ok=True)


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dt = torch.float32

full_ds = FirstDataset(h5_path)

generator = torch.Generator(device='cpu').manual_seed(42)

test_ds, _ = random_split(
    full_ds, [len(full_ds), 0], generator=generator)

test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                         num_workers=4, pin_memory=torch.cuda.is_available(),
                         persistent_workers=True, prefetch_factor=2,
                         collate_fn=collate_batch)

model = CPEN(levels=4, out_channels=8, device=dev, dtype=dt, DEBUG=True)
model = model.to(device=dev, dtype=dt)


model_fname = os.path.join(model_save_path, "best_model.pth")
state = torch.load(model_fname, map_location=dev)
model.load_state_dict(state)

idxs, RMSs = evaluate_one_epoch(model, test_loader, dev, dt)

print(f"Test RMS     : {np.mean(RMSs):.4f}")
print(f"Test RMS std : {np.std(RMSs):.4f}")
print(f"Test RMS min : {np.min(RMSs):.4f}")
print(f"Test RMS max : {np.max(RMSs):.4f}")

with open(os.path.join(model_save_path, "test_results_myYaleCroppedA.json"), "w") as f:
    json.dump({
        "test_rms_mean": np.mean(RMSs),
        "test_rms_std": np.std(RMSs),
        "test_rms_min": np.min(RMSs),
        "test_rms_max": np.max(RMSs),
        "test_idxs": idxs,
        "test_rms_values": RMSs
    }, f, indent=4)


with open(os.path.join(model_save_path, "test_results_myYaleCroppedA.json"), "r") as f:
    data = json.load(f)

rms_all = np.array(data["test_rms_values"], dtype=float)
test_idxs = np.array(data["test_idxs"], dtype=int)

sigma_counts = {str(i): 0 for i in range(1, 11)}
sigma_values = {str(i): [] for i in range(1, 11)}

for idx, value in zip(test_idxs, rms_all):
    s_id = str((idx // 10) % 10 + 1)
    sigma_counts[s_id] += 1
    sigma_values[s_id].append(float(value))

for s, count in sigma_counts.items():
    print(f"sigma {s:>2} : {count}")


max_rms = float(np.max(rms_all))
xmax_int = int(np.ceil(max_rms))
x_locator_1 = MultipleLocator(1)

bin_width = 0.5
bins = np.arange(0, xmax_int + bin_width, bin_width)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
xs = np.arange(1, 11)
ax.plot(xs, [sigma_counts[str(i)] for i in xs], marker="o")
ax.set_title("Πλήθος δειγμάτων ανά Sigma")
ax.set_xlabel("σ values")
ax.set_ylabel("Number of samples")
ax.set_xticks(xs)
ax.set_ylim(0, max(200, max(sigma_counts.values()) + 10))
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(fig_save_path,
            "number_of_samples_per_sigma.png"), dpi=200)


fig, (axh, axc) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

counts_all, _, _ = axh.hist(rms_all, bins=bins)
axh.set_title("Histogram of Total RMS Values")
axh.set_xlabel("RMS")
axh.set_ylabel("Count")
axh.xaxis.set_major_locator(x_locator_1)
axh.set_xlim(0, xmax_int)
axh.grid(True, alpha=0.25)

cdf_all = np.cumsum(counts_all) / np.sum(counts_all)
x_centers = (bins[:-1] + bins[1:]) / 2
axc.plot(x_centers, cdf_all)
axc.set_title("Cumulative Distribution (Overall)")
axc.set_xlabel("RMS")
axc.set_ylabel("CDF")
axc.xaxis.set_major_locator(x_locator_1)
axc.set_yticks(np.linspace(0, 1, 11))
axc.set_xlim(0, xmax_int)
axc.set_ylim(0, 1.0)
axc.grid(True, alpha=0.25)
plt.savefig(os.path.join(fig_save_path, "overall_histogram_n_cdf.png"), dpi=200)

fig, axes = plt.subplots(5, 2, figsize=(
    12, 12), sharex=True, sharey=True, constrained_layout=True)
axes = axes.ravel()

max_count = 0
for i in range(1, 11):
    vals = np.array(sigma_values[str(i)], dtype=float)
    if len(vals) == 0:
        continue
    c, _, _ = axes[i-1].hist(vals, bins=bins)
    max_count = max(max_count, np.max(c))
    axes[i-1].set_title(f"σ = {i}")
    axes[i-1].grid(True, alpha=0.25)

for ax in axes:
    ax.xaxis.set_major_locator(x_locator_1)
    ax.set_xlim(0, xmax_int)
    ax.set_ylim(0, max_count * 1.05 if max_count > 0 else 1)

fig.suptitle("Histogram ανά σ", fontsize=14)
fig.supxlabel("RMS")
fig.supylabel("Count")
plt.savefig(os.path.join(fig_save_path, "histograms_per_sigma.png"), dpi=200)

fig, axes = plt.subplots(5, 2, figsize=(
    12, 12), sharex=True, sharey=True, constrained_layout=True)
axes = axes.ravel()

for i in range(1, 11):
    vals = np.array(sigma_values[str(i)], dtype=float)
    if len(vals) == 0:
        axes[i-1].set_title(f"σ = {i} (no data)")
        axes[i-1].grid(True, alpha=0.25)
        continue

    counts, _ = np.histogram(vals, bins=bins)
    cdf = np.cumsum(
        counts) / np.sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts, dtype=float)
    axes[i-1].plot(x_centers, cdf)
    axes[i-1].set_title(f"σ = {i}")
    axes[i-1].grid(True, alpha=0.25)

for ax in axes:
    ax.xaxis.set_major_locator(x_locator_1)
    ax.set_xlim(0, xmax_int)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))

fig.suptitle("CDF ανά σ", fontsize=14)
fig.supxlabel("RMS")
fig.supylabel("CDF")
plt.savefig(os.path.join(fig_save_path, "cdf_per_samples.png"), dpi=200)
