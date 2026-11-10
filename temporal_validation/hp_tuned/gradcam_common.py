"""temporal_validation/hp_tuned/gradcam_common.py

Grad-CAM helper functions used by gradcam_resnet.py.

The function bodies are copied verbatim from the internal analysis script
(temporal_cohort_v2/04_gradcam.py, 2026-10-20); only the module wrapper differs.

- Target: class logit (pre-softmax). class 1 = FV<400 (dysfunction), class 0 = FV>=400.
- Class-average maps / frequency profiles are averaged per patient first, then across
  patients; bands are patient-level bootstrap 95% intervals.
- Image row 0 = lowest mel bin; figures use origin="lower".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASS_NAMES = {1: "FV<400", 0: "FV>=400"}
N_BOOT = 2000


# ===== 軸 =====

def row_frequencies(n_rows: int, n_mels: int, fmax: float) -> np.ndarray:
    """画像の各行 (0 = 下端/低周波) に対応する周波数 [Hz]（メルビン中心を補間）。"""
    centers = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=fmax, htk=False)[1:-1]
    mel_idx = (np.arange(n_rows) + 0.5) * n_mels / n_rows - 0.5
    return np.interp(mel_idx, np.arange(n_mels), centers)


def set_axes(ax, freqs: np.ndarray, n_cols: int, duration: float, fmax: float) -> None:
    ticks_hz = [f for f in [0, 100, 200, 400, 600, 800, 1000, 1500, 2000] if f <= fmax]
    ax.set_yticks(np.interp(ticks_hz, freqs, np.arange(len(freqs))))
    ax.set_yticklabels([str(t) for t in ticks_hz])
    xt = np.arange(0, duration + 1e-9, 1.0)
    ax.set_xticks(xt / duration * (n_cols - 1))
    ax.set_xticklabels([f"{t:.0f}" for t in xt])
    ax.set_ylabel("Hz"); ax.set_xlabel("s")


# ===== CAM 計算 =====

def load_images(paths: list, transform) -> tuple[torch.Tensor, np.ndarray]:
    tensors, grays = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
        grays.append(np.asarray(img)[:, :, 0].astype(np.float32) / 255.0)
    return torch.stack(tensors), np.stack(grays)


def compute_cams(model, layer, x: torch.Tensor, target_class: int, device, batch: int = 16) -> np.ndarray:
    out = []
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            out.append(cam(input_tensor=xb, targets=[ClassifierOutputTarget(target_class)] * len(xb)))
    return np.concatenate(out)


def patient_mean(arr: np.ndarray, patients: np.ndarray) -> tuple[np.ndarray, list]:
    ids = list(dict.fromkeys(patients))
    return np.stack([arr[patients == pid].mean(axis=0) for pid in ids]), ids


def bootstrap_band(per_patient: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    n = len(per_patient)
    boots = np.stack([per_patient[rng.integers(0, n, n)].mean(axis=0) for _ in range(N_BOOT)])
    return np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# ===== 図 =====

def fig_examples(sel: pd.DataFrame, gray, cams: dict, freqs, duration, fmax, out_path) -> None:
    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.6 * n), squeeze=False)
    for i, (_, r) in enumerate(sel.iterrows()):
        k = r["idx"]; c = int(r["pred_class"])
        panels = [(gray[k], None, "Mel spectrogram"),
                  (gray[k], cams[("layer4", c)][k], f"Grad-CAM layer4 → {CLASS_NAMES[c]}"),
                  (gray[k], cams[("layer3", c)][k], f"Grad-CAM layer3 → {CLASS_NAMES[c]}")]
        for j, (g, cam, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(g, cmap="gray", origin="lower", aspect="auto")
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.45, origin="lower", aspect="auto", vmin=0, vmax=1)
            set_axes(ax, freqs, g.shape[1], duration, fmax)
            ax.set_title(title, fontsize=8)
        axes[i, 0].set_ylabel(f"{r['category']}  {r['clip_file']}\n"
                              f"truth {CLASS_NAMES[int(r['true_label'])]}, p(rec)={r['rec_proba']:.2f}\nHz",
                              fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_class_average(avg: dict, freqs, duration, fmax, n_pat: dict, out_path, title: str) -> None:
    rows = [("spec", "Mean spectrogram"), ("layer4", "Grad-CAM layer4 (true class)"),
            ("layer3", "Grad-CAM layer3 (true class)")]
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    for i, (key, label) in enumerate(rows):
        for j, y in enumerate([1, 0]):
            ax = axes[i, j]
            if key == "spec":
                ax.imshow(avg[("spec", y)], cmap="gray", origin="lower", aspect="auto")
            else:
                ax.imshow(avg[("spec", y)], cmap="gray", origin="lower", aspect="auto")
                ax.imshow(avg[(key, y)], cmap="jet", alpha=0.5, origin="lower", aspect="auto", vmin=0, vmax=1)
            set_axes(ax, freqs, avg[("spec", y)].shape[1], duration, fmax)
            ax.set_title(f"{label}\n{CLASS_NAMES[y]} (n={n_pat[y]} patients)", fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_profiles(prof: dict, freqs, out_path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {1: "tab:red", 0: "tab:blue"}
    for ax, layer in zip(axes, ["layer4", "layer3"]):
        for y in [1, 0]:
            m, lo, hi, n = prof[(layer, y)]
            ax.plot(freqs, m, color=colors[y], label=f"{CLASS_NAMES[y]} → {CLASS_NAMES[y]} (n={n} pts)")
            ax.fill_between(freqs, lo, hi, color=colors[y], alpha=0.2)
        ax.set(xlabel="Frequency [Hz]", title=f"{layer}: Grad-CAM averaged over time", xlim=(0, freqs[-1]))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean Grad-CAM (0-1)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ===== main =====

