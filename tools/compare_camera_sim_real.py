#!/usr/bin/env python3
"""
实机相机 vs 仿真相机 成像相似度定量验收工具。

用法示例:
  python3 tools/compare_camera_sim_real.py --real real.jpg --sim sim.jpg
  python3 tools/compare_camera_sim_real.py --real real_dir/ --sim sim_dir/ --max-frames 50
  python3 tools/compare_camera_sim_real.py --real real.mp4 --sim sim.mp4 --max-frames 100
"""

import argparse
import glob
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_frames(path: str, max_frames: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if os.path.isdir(path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
        files.sort()
        for f in files[:max_frames]:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    elif os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
        else:
            cap = cv2.VideoCapture(path)
            while cap.isOpened() and len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
    return frames


def estimate_flat_noise_std(gray: np.ndarray) -> Tuple[float, float, float]:
    """用 Immerkaer 高通卷积核在低梯度平坦区域估计读出/总噪声标准差 (DN)。"""
    g = gray.astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = np.abs(cv2.filter2D(g, cv2.CV_32F, kernel))

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    flat_mask = grad < np.percentile(grad, 35)

    scale = math.sqrt(math.pi / 2.0) / 6.0
    total_std = float(np.mean(resp[flat_mask]) * scale) if np.any(flat_mask) else 0.0

    dark_mask = flat_mask & (gray < 60)
    bright_mask = flat_mask & (gray >= 100) & (gray <= 240)
    dark_std = float(np.mean(resp[dark_mask]) * scale) if np.sum(dark_mask) > 200 else total_std
    bright_std = float(np.mean(resp[bright_mask]) * scale) if np.sum(bright_mask) > 200 else total_std
    return total_std, dark_std, bright_std


def estimate_edge_sharpness(gray: np.ndarray) -> Tuple[float, float]:
    """返回 (Laplacian 方差, 强边缘平均过渡宽度 proxy: |grad_max| 归一化的倒数)。"""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    p99 = float(np.percentile(mag, 99.5))
    return lap_var, p99


def analyze_frame(bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    noise_all, noise_dark, noise_bright = estimate_flat_noise_std(gray)
    lap_var, edge_p995 = estimate_edge_sharpness(gray)

    # 中心 vs 四角亮度比（暗角 proxy，拍摄均匀墙面时最准）
    cy0, cy1 = int(h * 0.4), int(h * 0.6)
    cx0, cx1 = int(w * 0.4), int(w * 0.6)
    center_med = float(np.median(gray[cy0:cy1, cx0:cx1]))
    cw, ch = max(16, w // 10), max(16, h // 10)
    corners = np.concatenate([
        gray[:ch, :cw].ravel(),
        gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(),
        gray[-ch:, -cw:].ravel(),
    ])
    corner_med = float(np.median(corners))
    vignette_ratio = corner_med / max(1.0, center_med)

    sat_ratio = float(np.mean(gray >= 250)) * 100.0
    bright_mask = gray >= 150
    bright_sat_ratio = (
        float(np.mean(gray[bright_mask] >= 250)) * 100.0 if np.sum(bright_mask) > 50 else 0.0
    )

    return {
        "mean": float(np.mean(gray)),
        "p50": float(np.percentile(gray, 50)),
        "p95": float(np.percentile(gray, 95)),
        "p99": float(np.percentile(gray, 99)),
        "sat_pct": sat_ratio,
        "bright_sat_pct": bright_sat_ratio,
        "noise_all": noise_all,
        "noise_dark": noise_dark,
        "noise_bright": noise_bright,
        "lap_var": lap_var,
        "edge_p995": edge_p995,
        "vignette_ratio": vignette_ratio,
        "b_mean": float(np.mean(bgr[:, :, 0])),
        "g_mean": float(np.mean(bgr[:, :, 1])),
        "r_mean": float(np.mean(bgr[:, :, 2])),
    }


def summarize(frames: List[np.ndarray]) -> Tuple[Dict[str, float], np.ndarray]:
    metrics = [analyze_frame(f) for f in frames]
    keys = list(metrics[0].keys())
    agg = {k: float(np.median([m[k] for m in metrics])) for k in keys}

    hist = np.zeros(256, dtype=np.float64)
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
        hist += h
    hist /= max(1.0, hist.sum())
    return agg, hist


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    bc = float(np.sum(np.sqrt(np.maximum(0.0, h1 * h2))))
    return math.sqrt(max(0.0, 1.0 - min(1.0, bc)))


def main() -> int:
    parser = argparse.ArgumentParser(description="实机 vs 仿真相机成像相似度验收工具")
    parser.add_argument("--real", required=True, help="实机图像 / 目录 / 视频路径")
    parser.add_argument("--sim", required=True, help="仿真图像 / 目录 / 视频路径")
    parser.add_argument("--max-frames", type=int, default=50, help="最多采样帧数")
    args = parser.parse_args()

    real_frames = load_frames(args.real, args.max_frames)
    sim_frames = load_frames(args.sim, args.max_frames)
    if not real_frames:
        print(f"[ERROR] 无法读取实机样本: {args.real}", file=sys.stderr)
        return 1
    if not sim_frames:
        print(f"[ERROR] 无法读取仿真样本: {args.sim}", file=sys.stderr)
        return 1

    r_stat, r_hist = summarize(real_frames)
    s_stat, s_hist = summarize(sim_frames)
    bhatt = bhattacharyya(r_hist, s_hist)

    print(f"样本数: real={len(real_frames)} ({real_frames[0].shape[1]}x{real_frames[0].shape[0]}), "
          f"sim={len(sim_frames)} ({sim_frames[0].shape[1]}x{sim_frames[0].shape[0]})")
    print("-" * 74)
    print(f"{'指标':<24} {'实机 (Real)':>14} {'仿真 (Sim)':>14} {'差值 (Sim-Real)':>16}")
    print("-" * 74)
    rows = [
        ("灰度均值 (DN)", "mean"),
        ("灰度中位数 P50 (DN)", "p50"),
        ("灰度 P95 (DN)", "p95"),
        ("灰度 P99 (DN)", "p99"),
        ("全图过曝占比 >=250 (%)", "sat_pct"),
        ("高亮区过曝占比 (%)", "bright_sat_pct"),
        ("平坦区总噪声 std (DN)", "noise_all"),
        ("暗部噪声 std (DN)", "noise_dark"),
        ("亮部噪声 std (DN)", "noise_bright"),
        ("锐度 Laplacian 方差", "lap_var"),
        ("强边缘梯度 P99.5", "edge_p995"),
        ("四角/中心亮度比", "vignette_ratio"),
    ]
    for label, key in rows:
        rv, sv = r_stat[key], s_stat[key]
        print(f"{label:<24} {rv:>14.3f} {sv:>14.3f} {sv - rv:>+16.3f}")
    print("-" * 74)
    print(f"灰度直方图 Bhattacharyya 距离: {bhatt:.4f} (0=完全一致, <0.15 较接近)")

    print("\n调参建议（configs/sim.yaml -> sim_camera_model）:")
    if s_stat["noise_dark"] < r_stat["noise_dark"] - 0.3:
        print(f"  - 仿真暗部偏干净：可提高 read_noise_std 至 ~{r_stat['noise_dark']:.2f}")
    elif s_stat["noise_dark"] > r_stat["noise_dark"] + 0.3:
        print(f"  - 仿真暗部噪声偏大：可降低 read_noise_std 至 ~{r_stat['noise_dark']:.2f}")

    if s_stat["edge_p995"] > r_stat["edge_p995"] * 1.15:
        print("  - 仿真边缘偏锐利：可适当增大 gaussian_blur_sigma（步进 +0.1~0.2）")
    elif s_stat["edge_p995"] < r_stat["edge_p995"] * 0.85:
        print("  - 仿真边缘偏糊：可减小 gaussian_blur_sigma 或检查 exposure_ms")

    if s_stat["p95"] > 1.0 and abs(s_stat["p95"] - r_stat["p95"]) > 10.0:
        ratio = r_stat["p95"] / s_stat["p95"]
        print(f"  - 高亮亮度有偏差：建议按比例调整 brightness_scale（约乘以 {ratio:.2f}）")

    if s_stat["vignette_ratio"] > r_stat["vignette_ratio"] + 0.05:
        sug_vig = max(0.0, min(0.6, 1.0 - r_stat["vignette_ratio"]))
        print(f"  - 仿真四角偏亮（若为均匀背景对比）：可设 vignette_strength ≈ {sug_vig:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
