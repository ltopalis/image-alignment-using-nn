import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import h5py
import torch
import json
import math
import numpy as np
from matplotlib import pyplot as plt
from CPEN import CPEN
from matplotlib.ticker import MultipleLocator


save_fig_path = "./pretrained_models/images"
dataset_hdf5_file_path = "datasets/myYaleCroppedA/base/dataset_matlab.hdf5"
best_model_path = "pretrained_models/best_model.pth"
training_stats_path_json = "pretrained_models/train_results.json"
testing_stats_path_json = "pretrained_models/test_results.json"

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dt = torch.float32

os.makedirs(os.path.join(save_fig_path, 'testing'), exist_ok=True)
os.makedirs(os.path.join(save_fig_path, 'training'), exist_ok=True)
os.makedirs(os.path.join(save_fig_path, 'general'), exist_ok=True)

file = h5py.File(dataset_hdf5_file_path, "r")
img = torch.from_numpy(file['img'][0:10000:313]).float()
tmplt = torch.from_numpy(file['tmplt'][0:10000:313]).float()

# Visualize Dataset

fig, ax = plt.subplots(7, 8, figsize=(12, 9))

for a in ax.flat:
    a.axis("off")

for i in range(7):

    ax[i, 0].imshow(img[i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 0].axis("off")

    ax[i, 1].imshow(tmplt[i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 1].axis("off")

    ax[i, 2].imshow(img[7+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 2].axis("off")

    ax[i, 3].imshow(tmplt[7+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 3].axis("off")

    ax[i, 4].imshow(img[14+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 4].axis("off")

    ax[i, 5].imshow(tmplt[14+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 5].axis("off")

    ax[i, 6].imshow(img[21+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 6].axis("off")

    ax[i, 7].imshow(tmplt[21+i, :, :].squeeze().numpy(), cmap="gray")
    ax[i, 7].axis("off")


plt.tight_layout()
plt.savefig(os.path.join(save_fig_path, 'general', "dataset.png"), dpi=200)
plt.close()

# Visualize Feature Maps (image - no photometrical nor geometrical)

img = torch.from_numpy(file['img'][0]).float().to(device=dev)
tmplt = torch.from_numpy(file['tmplt'][0]).float().to(device=dev)

model = CPEN(levels=4, out_channels=8, device=dev, dtype=dt, DEBUG=False)
model.load_state_dict(torch.load(best_model_path,  map_location=dev))

with torch.no_grad():
    fm = model.model(img.unsqueeze(0).unsqueeze(0)).squeeze()
    fm_t = model.model(tmplt.unsqueeze(0).unsqueeze(0)).squeeze()

fig, ax = plt.subplots(3, 4, figsize=(12, 9))

for a in ax.flat:
    a.axis("off")

ax[0, 1].imshow(img.detach().cpu().numpy(), cmap="gray")
ax[0, 1].set_title("F0")
ax[0, 1].axis("off")

ax[0, 0].set_visible(False)
ax[0, 2].set_visible(False)
ax[0, 3].set_visible(False)

for i in range(4):
    ax[1, i].imshow(fm[i].detach().cpu().numpy(), cmap="gray")
    ax[1, i].set_title(f"F{i+1}")
    ax[1, i].axis("off")

for i in range(4):
    ax[2, i].imshow(fm[i+4].detach().cpu().numpy(), cmap="gray")
    ax[2, i].set_title(f"F{i+5}")
    ax[2, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_fig_path, 'general', "features.png"), dpi=200)
plt.close()

# Visualize Feature Maps (template - photometrical and geometrical)

fig, ax = plt.subplots(3, 4, figsize=(12, 9))

for a in ax.flat:
    a.axis("off")

ax[0, 1].imshow(tmplt.detach().cpu().numpy(), cmap="gray")
ax[0, 1].set_title("F0")
ax[0, 1].axis("off")

ax[0, 0].set_visible(False)
ax[0, 2].set_visible(False)
ax[0, 3].set_visible(False)

for i in range(4):
    ax[1, i].imshow(fm_t[i].detach().cpu().numpy(), cmap="gray")
    ax[1, i].set_title(f"F{i+1}")
    ax[1, i].axis("off")

for i in range(4):
    ax[2, i].imshow(fm_t[i+4].detach().cpu().numpy(), cmap="gray")
    ax[2, i].set_title(f"F{i+5}")
    ax[2, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_fig_path, "general", "features_t.png"), dpi=200)
plt.close()

# Visualize img + template -> result

img = torch.from_numpy(file['img'][0:100:10]
                       ).float().to(device=dev).unsqueeze(1)
tmplt = torch.from_numpy(file['tmplt'][0:100:10]).float().to(
    device=dev).unsqueeze(1)
init = torch.from_numpy(file['p_init'][0:100:10]).float().to(
    device=dev)

with torch.no_grad():
    pred = model(img, tmplt, init)
    temp = torch.eye(3).repeat(img.shape[0], 1, 1).to(device=dev)
    temp[:, :2, :] += pred
    pred = temp


def imwarp_affine_torch(img_bchw: torch.Tensor,
                        H_b33: torch.Tensor,
                        out_hw=None,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False):

    assert img_bchw.dim() == 4, "img_bchw must be [B,C,H,W]"
    B, C, H, W = img_bchw.shape

    if out_hw is None:
        H_out, W_out = H, W
    else:
        H_out, W_out = out_hw

    if H_b33.dim() == 2:
        H_b33 = H_b33.unsqueeze(0)
    assert H_b33.shape[0] == B and H_b33.shape[-2:] == (
        3, 3), "H_b33 must be [B,3,3]"

    device, dtype = img_bchw.device, img_bchw.dtype
    H_b33 = H_b33.to(device=device, dtype=dtype)

    def T_norm_to_pix(h, w):
        return torch.tensor([[(w - 1) / 2, 0, (w - 1) / 2],
                             [0, (h - 1) / 2, (h - 1) / 2],
                             [0, 0, 1]], device=device, dtype=dtype)

    def T_pix_to_norm(h, w):
        return torch.tensor([[2 / (w - 1), 0, -1],
                             [0, 2 / (h - 1), -1],
                             [0, 0, 1]], device=device, dtype=dtype)

    H_inv = torch.linalg.inv(H_b33)

    Tp2n_in = T_pix_to_norm(H, W)
    Tn2p_out = T_norm_to_pix(H_out, W_out)

    theta3 = Tp2n_in @ (H_inv @ Tn2p_out)
    theta3 = theta3.unsqueeze(0).expand(
        B, -1, -1) if theta3.dim() == 2 else theta3
    theta = theta3[:, :2, :]

    grid = F.affine_grid(theta, size=(B, C, H_out, W_out),
                         align_corners=align_corners)
    warped = F.grid_sample(img_bchw, grid, mode=mode,
                           padding_mode=padding_mode, align_corners=align_corners)
    return warped


with torch.no_grad():
    warped_img = imwarp_affine_torch(
        tmplt, pred, out_hw=img.shape[-2:])

B = warped_img.shape[0]

left_count = min(5, B)
right_count = max(0, B - 5)
rows = max(left_count, right_count)

fig = plt.figure(figsize=(14, 3.2 * rows), constrained_layout=False)

outer = fig.add_gridspec(
    rows,
    2,
    wspace=0.05,
    hspace=0.35
)


def plot_triplet(row, col, b):
    inner = outer[row, col].subgridspec(2, 3, height_ratios=[0.15, 1])

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.text(
        0.5, 0.5,
        f"σ = {b}",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold"
    )
    title_ax.axis("off")

    axes = [
        fig.add_subplot(inner[1, 0]),
        fig.add_subplot(inner[1, 1]),
        fig.add_subplot(inner[1, 2]),
    ]

    images = [
        tmplt[b, 0],
        img[b, 0],
        warped_img[b, 0],
    ]

    titles = ["Template", "Input", "Warped → Template"]

    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im.detach().cpu().numpy(), cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)


for i in range(left_count):
    plot_triplet(i, 0, i)

for i in range(right_count):
    plot_triplet(i, 1, i + 5)

fig.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.98,
    bottom=0.02
)

plt.savefig(
    os.path.join(save_fig_path, "general", "warp_check.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close()

# Visualize training stats (log10)

with open(training_stats_path_json, "r") as f:
    res_dict = json.load(f)


epoch_mean, epoch = [], []
for epoch_str, epoch_data in res_dict.items():
    dev_rms = epoch_data['dev']['rms']
    epoch_mean.append(np.mean(dev_rms))
    epoch.append(epoch_str)

epoch_mean_log = list(map(lambda x: math.log10(x), epoch_mean))

plt.figure()
plt.bar(
    range(1, len(epoch_mean_log)+1),
    epoch_mean,
    width=0.1,
    color='orange',
    edgecolor='black',
)
plt.ylabel("log10 RMS Error")
plt.xlabel("Epoch")
plt.xticks(range(1, len(epoch_mean_log) + 1, 2))
plt.savefig(os.path.join(save_fig_path, 'training', 'log_rms_error.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Visualize training stats

epoch = np.array(list(map(lambda x: int(x), epoch)))
epoch_mean = np.array(epoch_mean)

epoch_mean[4] = epoch_mean[6] = epoch_mean[8] = np.nan

valid = ~np.isnan(epoch_mean)
missing = np.isnan(epoch_mean)

plt.figure()
plt.bar(
    epoch[valid],
    epoch_mean[valid],
    width=0.1,
    color='orange',
    edgecolor='black',
)

interp = np.interp(epoch, epoch[valid], epoch_mean[valid])
plt.bar(
    epoch[missing],
    interp[missing],
    width=0.1,
    fill=False,
    edgecolor='red',
    linestyle='--',
    linewidth=2,
)
plt.ylim(2, 6)
plt.xticks(epoch[::2])
plt.xlabel('Epoch')
plt.ylabel('Mean Value')
plt.savefig(os.path.join(save_fig_path, 'training', 'rms_error.png'),
            dpi=300, bbox_inches='tight')

# Visualize Aggregator Weights

model.eval()
w = model.aggregator.logits.detach().cpu()
att = torch.softmax(w, dim=0)

top = torch.topk(att, k=min(10, att.numel()))
fig, ax = plt.subplots()
ax.axis('off')

table_data = [[c, v]
              for c, v in zip(top.indices.tolist(), map(lambda x: f"{x*100:.4}", top.values.tolist()))]

table = ax.table(
    cellText=table_data,
    colLabels=['Feature', 'Value (%)'],
    loc='center',
    cellLoc='center'
)

table.scale(1, 1.5)

plt.savefig(os.path.join(save_fig_path, 'general', 'channels_values.png'),
            dpi=300, bbox_inches='tight')
plt.close()


with open(testing_stats_path_json, "r") as f:
    data = json.load(f)

rms_all = np.array(data["test_rms_values"], dtype=float)
test_idxs = np.array(data["test_idxs"], dtype=int)

# Ομαδοποίηση ανά sigma (1..10)

sigma_counts = {str(i): 0 for i in range(1, 11)}
sigma_values = {str(i): [] for i in range(1, 11)}

for idx, value in zip(test_idxs, rms_all):
    s_id = str((idx // 10) % 10 + 1)
    sigma_counts[s_id] += 1
    sigma_values[s_id].append(float(value))

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
plt.savefig(os.path.join(save_fig_path, "testing",
            "number_of_samples_per_sigma.png"), dpi=200)

# Overall histogram + CDF
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
plt.savefig(os.path.join(save_fig_path, "testing",
            "overall_histogram_n_cdf.png"), dpi=200)


# Plot 3: 10 hist subplots (κοινά bins/όρια, x ανά 1)

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
plt.savefig(os.path.join(save_fig_path, "testing",
            "histograms_per_sigma.png"), dpi=200)


# Plot 4: 10 CDF subplots (x σε RMS ανά 1 σε ΟΛΑ)

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
plt.savefig(os.path.join(save_fig_path, "testing",
            "cdf_per_samples.png"), dpi=200)
