#!/usr/bin/env python3
"""
实机 ChArUco 无损标定 + 仿真相机 (SimCameraModel) 逐像素对齐验收与参数拟合工具。

原理：
  1. 用实机拍摄的无损 ChArUco 标定板照片，检测角点并做无损内参标定；
  2. 对每张图用 (K_real, D_real) 解出标定板精确 3D 位姿 (rvec, tvec)；
  3. 把程序生成的干净数字标定板按同一 (rvec, tvec) 投到 Bevy 仿真理想针孔 K_sim，
     再经过与 io::SimCameraModel 完全同构的几何重映射 + MTF模糊 + 光度 + 泊松-高斯噪声链路；
  4. 在标定板区域逐像素比对 I_real 与 I_sim，输出几何误差、锐度误差、噪声误差与推荐配置，
     并导出可视化残差对比图。
"""

import argparse
import glob
import math
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np


K_SIM_DEFAULT = np.array(
    [[1304.673, 0.0, 720.0], [0.0, 1304.673, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def make_charuco_board(squares_x: int, squares_y: int, marker_ratio: float, dict_id: int):
    adict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), 1.0, marker_ratio, adict)
        # 兼容旧版 OpenCV 生成的棋盘格布局（按需尝试）
    else:
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, 1.0, marker_ratio, adict)
    return board, adict


def detect_charuco(gray: np.ndarray, board, adict):
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 4

    def _run(img_in):
        if hasattr(cv2.aruco, "CharucoDetector"):
            det = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)
            return det.detectBoard(img_in)
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(img_in, adict, parameters=params)
        if m_ids is None or len(m_ids) < 4:
            return None, None, m_corners, m_ids
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, img_in, board)
        return ch_corners, ch_ids, m_corners, m_ids

    ch_corners, ch_ids, m_corners, m_ids = _run(gray)
    # 远距或暗光下用 CLAHE 辅助初检，亚像素精修仍在原始无损灰度图上进行
    if ch_ids is None or len(ch_ids) < 20:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g_enh = clahe.apply(gray)
        c2, i2, mc2, mi2 = _run(g_enh)
        if i2 is not None and (ch_ids is None or len(i2) > len(ch_ids)):
            ch_corners, ch_ids, m_corners, m_ids = c2, i2, mc2, mi2

    if ch_corners is not None and len(ch_corners) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        ch_corners = cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), criteria)
    return ch_corners, ch_ids, m_corners, m_ids


def detect_chessboard_fallback(gray: np.ndarray, squares_x: int, squares_y: int):
    """远距 ArUco 位太小检不出时，回退到标准棋盘格内角点检测（要求标定板大致正向摆放）。"""
    nx, ny = squares_x - 1, squares_y - 1
    # 先在半分辨率图上快速初检，避免全图高噪搜索耗时过长
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    s_norm = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, pts_s = cv2.findChessboardCorners(s_norm, (nx, ny), flags)
    if not ok or pts_s is None or len(pts_s) != nx * ny:
        return None, None
    pts = (pts_s * 2.0).astype(np.float32)
    # 保证行优先从左到右、从上到下，与 CharucoBoard.getChessboardCorners() 顺序一致
    if pts[0, 0, 0] > pts[nx - 1, 0, 0] and pts[0, 0, 1] > pts[-1, 0, 1]:
        pts = pts[::-1].copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
    ids = np.arange(nx * ny, dtype=np.int32).reshape(-1, 1)
    return pts, ids


def render_clean_board(board, squares_x: int, squares_y: int, px_per_sq: int = 200) -> np.ndarray:
    w = squares_x * px_per_sq
    h = squares_y * px_per_sq
    if hasattr(board, "generateImage"):
        img = board.generateImage((w, h), marginSize=0, borderBits=1)
    else:
        img = board.draw((w, h), marginSize=0, borderBits=1)
    return img


def build_sim_to_real_maps(
    out_w: int, out_h: int, K_sim: np.ndarray, K_real: np.ndarray, D_real: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2)
    src = cv2.undistortPoints(pts, K_real, D_real, R=np.eye(3), P=K_sim).reshape(out_h, out_w, 2)
    return src[:, :, 0].astype(np.float32), src[:, :, 1].astype(np.float32)


def board_homography_to_sim(
    rvec: np.ndarray, tvec: np.ndarray, K_sim: np.ndarray, px_per_sq: float
) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    # 标定板像素 (u_b, v_b) -> 板平面 3D 坐标 (u_b / px_per_sq, v_b / px_per_sq, 0)
    # P_cam = R * [x, y, 0]^T + t = [r1, r2, t] * [x, y, 1]^T
    M = np.column_stack([R[:, 0] / px_per_sq, R[:, 1] / px_per_sq, tvec.reshape(3)])
    H = K_sim @ M
    return H / H[2, 2]


def estimate_noise_on_flat_squares(
    real_gray: np.ndarray, clean_warped: np.ndarray, board_mask: np.ndarray
) -> Tuple[float, float, float, float]:
    """用 Immerkaer 高通滤波在棋盘格纯黑/纯白内部区域（远离边界与 ArUco 码）估计实机底噪。"""
    # 只取纯白 (>240) 或纯黑 (<15) 且腐蚀掉边缘 5px 后的纯色平坦区
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    black_flat = cv2.erode(((clean_warped < 15) & (board_mask > 0)).astype(np.uint8), k_erode) > 0
    white_flat = cv2.erode(((clean_warped > 240) & (board_mask > 0)).astype(np.uint8), k_erode) > 0

    g = real_gray.astype(np.float32)
    kern = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
    scale = math.sqrt(math.pi / 2.0) / 6.0

    b_mean = float(np.median(real_gray[black_flat])) if np.sum(black_flat) > 500 else 20.0
    w_mean = float(np.median(real_gray[white_flat])) if np.sum(white_flat) > 500 else 180.0
    b_std = float(np.mean(resp[black_flat]) * scale) if np.sum(black_flat) > 500 else 1.5
    w_std = float(np.mean(resp[white_flat]) * scale) if np.sum(white_flat) > 500 else 2.5
    return b_mean, w_mean, b_std, w_std


def main() -> int:
    parser = argparse.ArgumentParser(description="ChArUco 无损标定 + 仿真相机像素级误差验收")
    parser.add_argument("--dir", required=True, help="无损 PNG 目录（包含 pose1_0001.png 等）")
    parser.add_argument("--squares-x", type=int, default=14)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--out-vis", default="/tmp/sim_vs_real_charuco.png", help="残差对比图输出路径")
    args = parser.parse_args()

    # 每个 pose 取第 1 帧做标定与对齐
    pose_files = sorted(glob.glob(os.path.join(args.dir, "pose*_0001.png")))
    if not pose_files:
        pose_files = sorted(glob.glob(os.path.join(args.dir, "*.png")))[:10]
    if not pose_files:
        print(f"[ERROR] 未找到 PNG 文件: {args.dir}", file=sys.stderr)
        return 1

    print(f"加载 {len(pose_files)} 组无损姿态图: {[os.path.basename(f) for f in pose_files]}")
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in pose_files]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    H_img, W_img = grays[0].shape

    # 自动尝试 legacyPattern True/False 与字典
    best_setup = None
    for legacy in (False, True):
        for dict_name, dict_id in [
            ("5x5_250", cv2.aruco.DICT_5X5_250),
            ("5x5_100", cv2.aruco.DICT_5X5_100),
            ("4x4_250", cv2.aruco.DICT_4X4_250),
        ]:
            board, adict = make_charuco_board(args.squares_x, args.squares_y, 0.75, dict_id)
            if hasattr(board, "setLegacyPattern"):
                board.setLegacyPattern(legacy)
            ch_c, ch_i, _, _ = detect_charuco(grays[0], board, adict)
            n = 0 if ch_i is None else len(ch_i)
            if best_setup is None or n > best_setup[0]:
                best_setup = (n, legacy, dict_name, dict_id)

    n_det, legacy, dict_name, dict_id = best_setup
    print(f"自动识别标定板: {args.squares_x}x{args.squares_y}, dict={dict_name}, legacy={legacy}, 首帧角点数={n_det}")
    if n_det < 15:
        print("[ERROR] ChArUco 角点检出过少，请检查棋盘格行列数与字典", file=sys.stderr)
        return 1

    # Step 1: 无损 ChArUco 内参标定
    board0, adict = make_charuco_board(args.squares_x, args.squares_y, 0.75, dict_id)
    if hasattr(board0, "setLegacyPattern"):
        board0.setLegacyPattern(legacy)

    all_obj, all_img, valid_indices = [], [], []
    chess_corners_3d = board0.getChessboardCorners()
    for idx, g in enumerate(grays):
        ch_c, ch_i, _, _ = detect_charuco(g, board0, adict)
        src_tag = "ChArUco"
        if ch_i is None or len(ch_i) < 15:
            ch_c, ch_i = detect_chessboard_fallback(g, args.squares_x, args.squares_y)
            src_tag = "Chessboard回退"
        if ch_i is not None and len(ch_i) >= 15:
            obj_pts = chess_corners_3d[ch_i.ravel()].astype(np.float32)
            all_obj.append(obj_pts)
            all_img.append(ch_c.astype(np.float32))
            valid_indices.append(idx)
            print(f"  {os.path.basename(pose_files[idx])}: 检出 {len(ch_i)} 个角点 ({src_tag})")

    rmse, K_real, D_real, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, (W_img, H_img), None, None
    )
    print("\n=== 1. 无损 ChArUco 内参标定结果 ===")
    print(f"重投影 RMSE: {rmse:.4f} px (旧 JPEG 粗标为 1.49 px)")
    print(f"camera_matrix: {[round(float(x), 6) for x in K_real.ravel()]}")
    print(f"distort_coeffs: {[round(float(x), 8) for x in D_real.ravel()]}")

    # Step 2: 预计算 Sim -> Real 几何重映射表（与 C++ SimCameraModel 完全一致）
    map_x, map_y = build_sim_to_real_maps(W_img, H_img, K_SIM_DEFAULT, K_real, D_real)

    # 自动挑选最匹配实物打印比例的 marker_ratio（在 pose 0 上比对 MAE）
    px_per_sq = 160
    best_ratio, best_ratio_mae = 0.75, 1e9
    for ratio in (0.65, 0.70, 0.75, 0.78, 0.80):
        b_try, _ = make_charuco_board(args.squares_x, args.squares_y, ratio, dict_id)
        if hasattr(b_try, "setLegacyPattern"):
            b_try.setLegacyPattern(legacy)
        clean_b = render_clean_board(b_try, args.squares_x, args.squares_y, px_per_sq)
        H_mat = board_homography_to_sim(rvecs[0], tvecs[0], K_SIM_DEFAULT, px_per_sq)
        sim_raw = cv2.warpPerspective(clean_b, H_mat, (W_img, H_img), flags=cv2.INTER_AREA)
        sim_geom = cv2.remap(sim_raw, map_x, map_y, cv2.INTER_LINEAR)
        mask_raw = cv2.warpPerspective(
            np.full_like(clean_b, 255), H_mat, (W_img, H_img), flags=cv2.INTER_NEAREST
        )
        mask = cv2.erode(cv2.remap(mask_raw, map_x, map_y, cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)) > 127
        # 简单线性对齐后比 MAE
        g0 = grays[valid_indices[0]].astype(np.float32)
        s0 = sim_geom.astype(np.float32)
        a = np.std(g0[mask]) / max(1e-3, np.std(s0[mask]))
        b = np.mean(g0[mask]) - a * np.mean(s0[mask])
        mae = float(np.mean(np.abs(g0[mask] - (a * s0[mask] + b))))
        if mae < best_ratio_mae:
            best_ratio_mae = mae
            best_ratio = ratio

    print(f"\n自动匹配打印 ArUco 比例 marker_ratio={best_ratio:.2f} (粗对齐 MAE={best_ratio_mae:.2f} DN)")
    board, _ = make_charuco_board(args.squares_x, args.squares_y, best_ratio, dict_id)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(legacy)
    clean_board = render_clean_board(board, args.squares_x, args.squares_y, px_per_sq)

    # Step 3: 对所有姿态生成理想仿真图 -> 几何重映射，统计底噪与拟合 MTF/光度参数
    geom_pairs = []
    b_means, w_means, b_stds, w_stds = [], [], [], []
    for k, idx in enumerate(valid_indices):
        H_mat = board_homography_to_sim(rvecs[k], tvecs[k], K_SIM_DEFAULT, px_per_sq)
        sim_raw = cv2.warpPerspective(clean_board, H_mat, (W_img, H_img), flags=cv2.INTER_AREA)
        sim_geom = cv2.remap(sim_raw, map_x, map_y, cv2.INTER_LINEAR)
        mask_raw = cv2.warpPerspective(
            np.full_like(clean_board, 255), H_mat, (W_img, H_img), flags=cv2.INTER_NEAREST
        )
        mask = cv2.erode(cv2.remap(mask_raw, map_x, map_y, cv2.INTER_NEAREST), np.ones((9, 9), np.uint8)) > 127
        bm, wm, bs, ws = estimate_noise_on_flat_squares(grays[idx], sim_geom, mask.astype(np.uint8) * 255)
        b_means.append(bm)
        w_means.append(wm)
        b_stds.append(bs)
        w_stds.append(ws)
        geom_pairs.append((idx, grays[idx], sim_geom, mask))

    # 拟合泊松-高斯噪声: Var(I) = read_std^2 + shot_k * I
    I_b = float(np.median(b_means))
    I_w = float(np.median(w_means))
    var_b = float(np.median(b_stds)) ** 2
    var_w = float(np.median(w_stds)) ** 2
    shot_k = max(0.0, (var_w - var_b) / max(1.0, I_w - I_b))
    read_std = math.sqrt(max(0.0, var_b - shot_k * I_b))

    # 网格搜索最佳 gaussian_blur_sigma（并记录各姿态各自的局部最佳 sigma，区分近距离焦与远距合焦）
    sigma_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5]
    best_sigma, best_mae = 0.0, 1e9
    sigma_table = []
    per_pose_best_sigma = {}
    for idx, g_real, s_geom, mask in geom_pairs:
        p_best_s, p_best_m = 0.0, 1e9
        for sigma in sigma_grid:
            s_blur = cv2.GaussianBlur(s_geom, (0, 0), sigma) if sigma > 0.05 else s_geom
            x = s_blur[mask].astype(np.float64)
            y = g_real[mask].astype(np.float64)
            A = np.column_stack([x, np.ones_like(x)])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = np.clip(coef[0] * x + coef[1], 0, 255)
            m = float(np.mean(np.abs(y - pred)))
            if m < p_best_m:
                p_best_m, p_best_s = m, sigma
        per_pose_best_sigma[idx] = (p_best_s, p_best_m)

    for sigma in sigma_grid:
        maes = []
        for idx, g_real, s_geom, mask in geom_pairs:
            s_blur = cv2.GaussianBlur(s_geom, (0, 0), sigma) if sigma > 0.05 else s_geom
            # 最小二乘对齐当前姿态的纸面照度 (scale, offset)
            x = s_blur[mask].astype(np.float64)
            y = g_real[mask].astype(np.float64)
            A = np.column_stack([x, np.ones_like(x)])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = np.clip(coef[0] * x + coef[1], 0, 255)
            maes.append(float(np.mean(np.abs(y - pred))))
        mean_mae = float(np.mean(maes))
        sigma_table.append((sigma, mean_mae))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_sigma = sigma

    # Step 4: 用拟合出的 SimCameraModel 参数生成完整退化仿真图，逐项验收
    rng = np.random.default_rng(42)
    corner_errs = []
    edge_ratios = []
    sim_b_stds, sim_w_stds = [], []
    vis_saved = False

    print("\n=== 2. 仿真相机 vs 实机相机 逐姿态对齐验收 ===")
    for p_i, (idx, g_real, s_geom, mask) in enumerate(geom_pairs):
        s_blur = cv2.GaussianBlur(s_geom, (0, 0), best_sigma) if best_sigma > 0.05 else s_geom.copy()
        x = s_blur[mask].astype(np.float64)
        y = g_real[mask].astype(np.float64)
        A = np.column_stack([x, np.ones_like(x)])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        sim_photo = np.clip(coef[0] * s_blur.astype(np.float64) + coef[1], 0.0, 255.0)

        # 叠加拟合出的泊松-高斯噪声
        std_map = np.sqrt(np.maximum(0.0, read_std * read_std + shot_k * sim_photo))
        sim_full = np.clip(sim_photo + rng.normal(0.0, 1.0, sim_photo.shape) * std_map, 0, 255).astype(
            np.uint8
        )

        # (a) 几何误差：在 sim_full 与 g_real 上分别跑角点检测，比对同一角点 ID 的像素差
        c_r, id_r, _, _ = detect_charuco(g_real, board, adict)
        c_s, id_s, _, _ = detect_charuco(sim_full, board, adict)
        if id_r is None or len(id_r) < 15 or id_s is None or len(id_s) < 15:
            c_r, id_r = detect_chessboard_fallback(g_real, args.squares_x, args.squares_y)
            c_s, id_s = detect_chessboard_fallback(sim_full, args.squares_x, args.squares_y)
        p_err = []
        if id_r is not None and id_s is not None:
            map_r = {int(i): c_r[j, 0] for j, i in enumerate(id_r.ravel())}
            for j, i in enumerate(id_s.ravel()):
                if int(i) in map_r:
                    p_err.append(float(np.linalg.norm(c_s[j, 0] - map_r[int(i)])))
        corner_errs.extend(p_err)

        # (b) 边缘锐度比值 (P99 梯度)
        def grad_p99(im):
            gx = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
            return float(np.percentile(np.hypot(gx, gy)[mask], 99.0))

        gr_r = grad_p99(g_real)
        gr_s = grad_p99(sim_full)
        edge_ratios.append(gr_s / max(1e-3, gr_r))

        # (c) 仿真图自身的平坦区噪声
        _, _, sbs, sws = estimate_noise_on_flat_squares(sim_full, s_geom, mask.astype(np.uint8) * 255)
        sim_b_stds.append(sbs)
        sim_w_stds.append(sws)

        mae_full = float(np.mean(np.abs(g_real[mask].astype(np.float32) - sim_full[mask].astype(np.float32))))
        rmse_full = float(
            np.sqrt(np.mean((g_real[mask].astype(np.float32) - sim_full[mask].astype(np.float32)) ** 2))
        )
        p_sig, p_mae_opt = per_pose_best_sigma[idx]
        print(
            f"  [{os.path.basename(pose_files[idx])}] 角点均值误差={np.mean(p_err) if p_err else -1:.3f}px "
            f"(N={len(p_err)}) | 像素MAE={mae_full:.2f} DN, RMSE={rmse_full:.2f} DN | "
            f"边缘梯度比(Sim/Real)={gr_s/max(1e-3,gr_r):.3f} | 该姿态最佳sigma={p_sig:.1f}px (MAE={p_mae_opt:.2f})"
        )

        if not vis_saved:
            vis_saved = True
            ys, xs = np.where(mask)
            y0, y1 = max(0, ys.min() - 20), min(H_img, ys.max() + 20)
            x0, x1 = max(0, xs.min() - 20), min(W_img, xs.max() + 20)
            crop_r = cv2.cvtColor(g_real[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            crop_s = cv2.cvtColor(sim_full[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            diff = np.abs(g_real[y0:y1, x0:x1].astype(np.int16) - sim_full[y0:y1, x0:x1].astype(np.int16))
            diff_vis = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            diff_vis[~mask[y0:y1, x0:x1]] = 0
            cv2.putText(crop_r, "Real (Lossless PNG)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(
                crop_s,
                f"SimCameraModel (sigma={best_sigma:.1f})",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(diff_vis, "|Real - Sim| x4 (board)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panel = np.hstack([crop_r, crop_s, diff_vis])
            cv2.imwrite(args.out_vis, panel)

    print("\n=== 3. 汇总指标与推荐 sim_camera_model 参数 ===")
    print(f"几何对齐：共 {len(corner_errs)} 个角点，均值误差={np.mean(corner_errs):.3f} px，P95={np.percentile(corner_errs, 95):.3f} px")
    print(f"镜头 MTF：最佳 gaussian_blur_sigma = {best_sigma:.2f} px (无模糊 MAE={sigma_table[0][1]:.2f} DN -> 拟合后 MAE={best_mae:.2f} DN)")
    print(f"边缘锐度比 (Sim/Real P99 梯度): {np.mean(edge_ratios):.3f} (1.000 为完全一致)")
    print(f"黑格噪声 std (I≈{I_b:.0f}): Real={np.median(b_stds):.2f} DN vs Sim={np.median(sim_b_stds):.2f} DN")
    print(f"白格噪声 std (I≈{I_w:.0f}): Real={np.median(w_stds):.2f} DN vs Sim={np.median(sim_w_stds):.2f} DN")
    print(f"推荐参数：")
    print(f"  gaussian_blur_sigma: {best_sigma:.2f}")
    print(f"  read_noise_std: {read_std:.2f}")
    print(f"  shot_noise_k: {shot_k:.4f}")
    print(f"  black_level_offset: {max(0.0, I_b - 5.0):.1f}  # 视场景环境光微调")
    print(f"对比图已保存至: {args.out_vis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
