"""temporal_validation/hp_tuned/gradcam_resnet.py

v4 追加（2026-10-27）: HP調整ありの ResNet200d 最終モデル（retrain_final_seeded.py）について、
temporal cohort の画像に Grad-CAM をかける。

- 計算・作図の関数は gradcam_common.py（2026-10-20 版 04_gradcam.py の関数をそのまま移したもの）を使う。
  違いは対象モデルと画像だけ。
- 画像は --images で選ぶ: noisy（主解析、evaluate_external_noisy.py の画像）/ clean（感度分析、
  evaluate_external.py の画像）。
- 音源の TP/TN/FP/FN は、開発データからロックした音源レベル閾値で判定する。
- 時間軸は解釈しない（10/20 の時間シフトテストで、時間方向の局在は CNN の位置バイアスと判明）。
  周波数方向のプロファイルだけを結果として使う。

実行（このフォルダで、analyze_external.py の後に）:
    uv run gradcam_resnet.py --images noisy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import gradcam_common as gc  # noqa: E402  (Grad-CAM helper functions)

from tune_and_develop import build_condition_name, build_runtime_config, load_config  # noqa: E402
from utils.dataset import build_transform  # noqa: E402
from utils.metadata_io import read_clip_metadata  # noqa: E402
from utils.trainer import create_model  # noqa: E402

ARCH = "resnet200d"
N_EXAMPLES = 2
SUBDIRS = {"noisy": "external_noisy", "clean": "external"}
IMAGE_ROOTS = {"noisy": "./images_external_noisy", "clean": None}  # clean は config の external.images_root


def main(images: str) -> int:
    config = load_config("config_external.yaml")
    sc = config["selected_condition"]
    condition = build_condition_name(sc)
    duration = config["audio"]["clip_duration_sec"]
    arch_dir = Path(config["results_root"]) / ARCH / condition
    out_dir = arch_dir / f"gradcam_{images}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    runtime_config = build_runtime_config(config, ARCH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(arch_dir / "final_model.pth", map_location="cpu", weights_only=False)
    model_cfg = dict(runtime_config["model"]["main"], pretrained=False)
    model = create_model(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    layers = {"layer4": model.layer4, "layer3": model.layer3}
    transform = build_transform("test", runtime_config)

    with open(arch_dir / "locked_threshold.json", encoding="utf-8") as f:
        locked_thr = json.load(f)["recording_threshold"]
    ext_meta = read_clip_metadata(config["external"]["clip_metadata_csv"])
    rec = pd.read_csv(arch_dir / SUBDIRS[images] / "external_predictions_recording.csv")  # analyze_external.py の出力
    df = ext_meta.merge(rec[["audio_file", "pred_proba_mean"]].rename(columns={"pred_proba_mean": "rec_proba"}),
                        on="audio_file").reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    df["pred_class"] = (df["rec_proba"] >= locked_thr).astype(int)
    df["category"] = np.select(
        [(df.label == 1) & (df.pred_class == 1), (df.label == 0) & (df.pred_class == 0),
         (df.label == 0) & (df.pred_class == 1), (df.label == 1) & (df.pred_class == 0)],
        ["TP", "TN", "FP", "FN"], default="NA")
    df = df.rename(columns={"label": "true_label"})

    root = Path(IMAGE_ROOTS[images] or config["external"]["images_root"]) / condition
    paths = [root / ("positive" if r.true_label == 1 else "negative") / f"{Path(r.clip_file).stem}.png"
             for r in df.itertuples()]
    x, gray = gc.load_images(paths, transform)

    cams = {(ln, c): gc.compute_cams(model, layer, x, c, device)
            for ln, layer in layers.items() for c in (0, 1)}
    np.savez_compressed(out_dir / "cams.npz", clip_file=df["clip_file"].values,
                        **{f"{ln}_class{c}": v for (ln, c), v in cams.items()})
    df.drop(columns=["idx"]).to_csv(out_dir / "clip_categories.csv", index=False)

    freqs = gc.row_frequencies(gray.shape[1], sc["n_mels"], sc["fmax"])
    patients = df["patient_id"].astype(str).values

    picks = []
    for cat, asc in [("TP", False), ("TN", True), ("FP", False), ("FN", True)]:
        sub = df[df.category == cat].sort_values("rec_proba", ascending=asc)
        picks.append(sub.drop_duplicates("patient_id").head(N_EXAMPLES))
    sel = pd.concat(picks)
    if len(sel):
        gc.fig_examples(sel, gray, cams, freqs, duration, sc["fmax"], out_dir / "examples_by_category.png")

    avg, n_pat, prof = {}, {}, {}
    for y in (1, 0):
        mask = df["true_label"].values == y
        pm, ids = gc.patient_mean(gray[mask], patients[mask])
        avg[("spec", y)] = pm.mean(axis=0)
        n_pat[y] = len(ids)
        for ln in layers:
            pm_cam, _ = gc.patient_mean(cams[(ln, y)][mask], patients[mask])
            avg[(ln, y)] = pm_cam.mean(axis=0)
            per_pat_profile = pm_cam.mean(axis=2)
            lo, hi = gc.bootstrap_band(per_pat_profile, rng)
            prof[(ln, y)] = (per_pat_profile.mean(axis=0), lo, hi, len(ids))
    title = f"ResNet200d (HP-tuned), temporal cohort ({images} images), {condition}"
    gc.fig_class_average(avg, freqs, duration, sc["fmax"], n_pat, out_dir / "class_average.png", title)
    gc.fig_profiles(prof, freqs, out_dir / "frequency_profile.png", title)

    prof_rows = []
    for (ln, y), (m, lo, hi, n) in prof.items():
        prof_rows.append({"layer": ln, "class": gc.CLASS_NAMES[y], "n_patients": n,
                          "peak_hz": float(freqs[int(np.argmax(m))]),
                          "share_below_300hz": float(m[freqs < 300].sum() / m.sum()),
                          "share_300_600hz": float(m[(freqs >= 300) & (freqs < 600)].sum() / m.sum()),
                          "share_above_600hz": float(m[freqs >= 600].sum() / m.sum())})
        pd.DataFrame({"freq_hz": freqs, "mean": m, "ci_lower": lo, "ci_upper": hi}).to_csv(
            out_dir / f"profile_{ln}_class{y}.csv", index=False)
    summary = {"locked_threshold": locked_thr,
               "categories": df.drop_duplicates("audio_file")["category"].value_counts().to_dict(),
               "profiles": prof_rows}
    print(f"categories (recordings): {summary['categories']}")
    print(pd.DataFrame(prof_rows).round(3).to_string(index=False))

    # サニティチェック: パラメータのランダム化（各クラス最大5クリップ、患者重複なし）
    sub = pd.concat([df[df.true_label == y].drop_duplicates("patient_id").head(5) for y in (1, 0)])
    xs = x[sub["idx"].values]
    trained = {c: cams[("layer4", c)][sub["idx"].values] for c in (0, 1)}

    def spearman_vs(rand_model) -> float:
        rs = []
        for c in (0, 1):
            rc = gc.compute_cams(rand_model, rand_model.layer4, xs, c, device)
            for a, b in zip(trained[c], rc):
                if a.std() > 0 and b.std() > 0:
                    rs.append(spearmanr(a.ravel(), b.ravel()).correlation)
        return float(np.mean(rs)) if rs else float("nan")

    torch.manual_seed(0)
    m_top = create_model(model_cfg, device)
    m_top.load_state_dict(ckpt["model_state_dict"])
    for mod in list(m_top.layer4.modules()) + [m_top.get_classifier()]:
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    m_top.eval()
    m_full = create_model(model_cfg, device).eval()
    summary["sanity_randomization"] = {
        "n_clips": int(len(sub)),
        "spearman_trained_vs_layer4+fc_randomized": spearman_vs(m_top),
        "spearman_trained_vs_fully_random": spearman_vs(m_full),
        "note": "低い相関 (|r|<~0.3) ほど、CAM が学習済み重みに依存している (妥当)",
    }
    print("sanity:", summary["sanity_randomization"])

    with open(out_dir / "gradcam_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", choices=list(SUBDIRS), default="noisy")
    sys.exit(main(parser.parse_args().images))
