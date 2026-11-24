#!/usr/bin/env python3
"""
端到端云台传递函数辨识脚本（纯 Python 3 标准库，无 numpy/scipy 依赖）。
读取 sim_gimbal_sysid 导出的 CSV，分别对 Yaw / Pitch 拟合：
  1) 一阶 + 纯延迟: G1(s) = exp(-tau_d * s) / (T * s + 1)
  2) 二阶 + 纯延迟: G2(s) = wn^2 * exp(-tau_d * s) / (s^2 + 2*zeta*wn*s + wn^2)
输出纯延迟、带宽、4.7 Hz（典型小陀螺换板频率）相位滞后、拟合 R^2 与 YAML 参数建议。
"""

import argparse
import cmath
import csv
import math


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t_sim_s"]),
                "phase": r["phase"],
                "uy": float(r["cmd_yaw_rad"]),
                "yy": float(r["muzzle_yaw_rad"]),
                "up": float(r["cmd_pitch_rad"]),
                "yp": float(r["muzzle_pitch_rad"]),
                "uvy": float(r["cmd_yaw_vel"]) if r.get("cmd_yaw_vel") else None,
                "uvp": float(r["cmd_pitch_vel"]) if r.get("cmd_pitch_vel") else None,
                "vy": float(r["yaw_vel_meas"]) if r.get("yaw_vel_meas") else None,
                "vp": float(r["pitch_vel_meas"]) if r.get("pitch_vel_meas") else None,
                "bs": float(r["bullet_speed"]) if r.get("bullet_speed") else None,
            })
    return rows


def resample_series(ts, vals, dt=0.005):
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) / dt) + 1
    out = []
    j = 0
    for i in range(n):
        t = t0 + i * dt
        while j + 1 < len(ts) and ts[j + 1] < t:
            j += 1
        if j + 1 < len(ts) and ts[j + 1] > ts[j]:
            a = (t - ts[j]) / (ts[j + 1] - ts[j])
            v = vals[j] + a * (vals[j + 1] - vals[j])
        else:
            v = vals[j]
        out.append(v)
    return out


def resample_uniform(ts, us, ys, dt=0.005):
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) / dt) + 1
    out_t, out_u, out_y = [], [], []
    j = 0
    for i in range(n):
        t = t0 + i * dt
        while j + 1 < len(ts) and ts[j + 1] < t:
            j += 1
        if j + 1 < len(ts) and ts[j + 1] > ts[j]:
            a = (t - ts[j]) / (ts[j + 1] - ts[j])
            u = us[j] + a * (us[j + 1] - us[j])
            y = ys[j] + a * (ys[j + 1] - ys[j])
        else:
            u, y = us[j], ys[j]
        out_t.append(t)
        out_u.append(u)
        out_y.append(y)
    return out_t, out_u, out_y


def sim_first_order(u, dt, delay_steps, tau):
    n = len(u)
    y = [0.0] * n
    a = math.exp(-dt / max(1e-5, tau))
    b = 1.0 - a
    cur = u[0]
    for i in range(n):
        idx = i - delay_steps
        ui = u[idx] if idx >= 0 else u[0]
        cur = a * cur + b * ui
        y[i] = cur
    return y


def sim_second_order_pva(u_pos, u_vel, dt, delay_steps, wn, zeta, ff_ratio=0.0):
    """
    二阶 + 速度前馈 + 纯延迟模型：
      y_ddot + 2*zeta*wn*y_dot + wn^2*y = wn^2*u_pos(t-tau) + (ff_ratio*2*zeta*wn)*u_vel(t-tau)
    返回 (y_pos, y_vel) 双通道输出，用于联合拟合 (角度, 角速度)。
    """
    n = len(u_pos)
    y_pos = [0.0] * n
    y_vel = [0.0] * n
    pos = u_pos[0]
    vel = 0.0
    wn2 = wn * wn
    c = 2.0 * zeta * wn
    kv = ff_ratio * c
    sub = max(1, int(math.ceil(dt * wn * max(1.0, 2.0 * zeta) / 0.15)))
    h = dt / sub
    for i in range(n):
        idx = i - delay_steps
        up = u_pos[idx] if idx >= 0 else u_pos[0]
        uv = (u_vel[idx] if idx >= 0 else 0.0) if u_vel is not None else 0.0
        for _ in range(sub):
            acc1 = wn2 * (up - pos) + kv * uv - c * vel
            p_mid = pos + 0.5 * h * vel
            v_mid = vel + 0.5 * h * acc1
            acc2 = wn2 * (up - p_mid) + kv * uv - c * v_mid
            pos += h * v_mid
            vel += h * acc2
        y_pos[i] = pos
        y_vel[i] = vel
    return y_pos, y_vel


def sim_second_order(u, dt, delay_steps, wn, zeta):
    yp, _ = sim_second_order_pva(u, None, dt, delay_steps, wn, zeta, 0.0)
    return yp


def mse_and_r2(y_meas, y_pred, start_idx=0):
    ys = y_meas[start_idx:]
    yp = y_pred[start_idx:]
    n = len(ys)
    mean_y = sum(ys) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(ys, yp))
    ss_tot = sum((a - mean_y) ** 2 for a in ys) + 1e-12
    rmse = math.sqrt(ss_res / n)
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2


def estimate_kinematic_limits(u, y, dt, step_u_ref=None):
    """从平滑重采样序列估计云台最大角速度、有效最大角加速度（反映惯量）与阶跃超调。"""
    # 5 点移动平均抑制高频编码器/IMU 微分噪声
    n = len(y)
    ys = [sum(y[max(0, i - 2):min(n, i + 3)]) / (min(n, i + 3) - max(0, i - 2)) for i in range(n)]
    vel = [(ys[i + 1] - ys[i - 1]) / (2.0 * dt) for i in range(1, n - 1)]
    acc = [(vel[i + 1] - vel[i - 1]) / (2.0 * dt) for i in range(1, len(vel) - 1)]
    abs_v = sorted(abs(v) for v in vel)
    abs_a = sorted(abs(a) for a in acc)
    max_vel = abs_v[int(0.99 * (len(abs_v) - 1))] if abs_v else 0.0
    max_acc = abs_a[int(0.98 * (len(abs_a) - 1))] if abs_a else 0.0

    # 阶跃超调（相对于指令峰值幅值）
    u_peak = max(abs(x) for x in u) if u else 0.0
    y_peak = max(abs(x) for x in y) if y else 0.0
    overshoot_pct = max(0.0, (y_peak - u_peak) / u_peak * 100.0) if u_peak > 1e-3 else 0.0
    return max_vel, max_acc, overshoot_pct


def fit_axis(name, rows, u_key, y_key, phases, dt=0.005, uv_key=None, vy_key=None):
    sub = [r for r in rows if r["phase"] in phases]
    if len(sub) < 30:
        print(f"[{name}] 有效样本不足 ({len(sub)})，跳过")
        return None

    ts = [r["t"] for r in sub]
    us = [r[u_key] for r in sub]
    ys = [r[y_key] for r in sub]
    ut, u, y = resample_uniform(ts, us, ys, dt=dt)

    has_uv = uv_key is not None and any(r[uv_key] is not None and abs(r[uv_key]) > 1e-4 for r in sub)
    has_vy = vy_key is not None and any(r[vy_key] is not None and abs(r[vy_key]) > 1e-4 for r in sub)
    u_vel = resample_series(ts, [r[uv_key] or 0.0 for r in sub], dt=dt) if has_uv else None
    y_vel_meas = resample_series(ts, [r[vy_key] or 0.0 for r in sub], dt=dt) if has_vy else None

    skip = int(0.1 / dt)
    y_span = max(y[skip:]) - min(y[skip:])
    if y_span < 0.01:
        print(f"\n[{name}] 警告：实测云台幅值仅 {math.degrees(y_span):.3f} deg，几乎未响应指令（请确认仿真已按 F5 开启外部接管）")
        return None

    # 速度归一化权重：把角速度 RMSE 折算到与角度 RMSE 同量纲联合优化
    pos_scale = max(1e-3, max(abs(x) for x in y[skip:]))
    vel_scale = max(1e-2, max(abs(x) for x in y_vel_meas[skip:])) if has_vy else 1.0

    def joint_loss(yp, yv):
        rp, _ = mse_and_r2(y, yp, skip)
        if not has_vy:
            return rp
        rv, _ = mse_and_r2(y_vel_meas, yv, skip)
        return rp + 0.5 * (pos_scale / vel_scale) * rv

    # 1) 一阶 + 纯延迟 网格搜索 + 细化
    best1 = (1e9, 0, 0.05)
    for d in range(0, int(0.16 / dt) + 1, 2):
        for tau in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]:
            yp = sim_first_order(u, dt, d, tau)
            rmse, _ = mse_and_r2(y, yp, skip)
            if rmse < best1[0]:
                best1 = (rmse, d, tau)
    # 细化
    _, d0, tau0 = best1
    for d in range(max(0, d0 - 3), d0 + 4):
        for k in range(-10, 11):
            tau = max(0.001, tau0 * (1.0 + 0.05 * k))
            yp = sim_first_order(u, dt, d, tau)
            rmse, _ = mse_and_r2(y, yp, skip)
            if rmse < best1[0]:
                best1 = (rmse, d, tau)
    rmse1, d1, tau1 = best1
    _, r2_1 = mse_and_r2(y, sim_first_order(u, dt, d1, tau1), skip)

    # 2) 二阶 (+ 速度前馈零点) + 纯延迟 网格搜索 + 坐标细化
    ff_grid = [0.0, 0.3, 0.6, 1.0] if has_uv else [0.0]
    best2 = (1e9, d1, 30.0, 0.8, 0.0)
    for d in range(max(0, d1 - 6), d1 + 7, 2):
        for wn in [12.0, 20.0, 30.0, 45.0, 65.0, 90.0]:
            for zeta in [0.4, 0.6, 0.8, 1.0, 1.3]:
                for ff in ff_grid:
                    yp, yv = sim_second_order_pva(u, u_vel, dt, d, wn, zeta, ff)
                    loss = joint_loss(yp, yv)
                    if loss < best2[0]:
                        best2 = (loss, d, wn, zeta, ff)
    # 细化
    for step_scale in [0.25, 0.08, 0.025]:
        improved = True
        while improved:
            improved = False
            _, cd, cwn, cz, cff = best2
            ff_deltas = (-step_scale, 0.0, step_scale) if has_uv else (0.0,)
            for dd in (-1, 0, 1):
                nd = max(0, cd + dd)
                for dw in (-step_scale, 0.0, step_scale):
                    nwn = min(250.0, max(2.0, cwn * (1.0 + dw)))
                    for dz in (-step_scale, 0.0, step_scale):
                        nz = min(3.0, max(0.15, cz * (1.0 + dz)))
                        for dff in ff_deltas:
                            nff = min(1.5, max(0.0, cff + dff))
                            if dd == 0 and dw == 0 and dz == 0 and dff == 0:
                                continue
                            yp, yv = sim_second_order_pva(u, u_vel, dt, nd, nwn, nz, nff)
                            loss = joint_loss(yp, yv)
                            if loss < best2[0] - 1e-7:
                                best2 = (loss, nd, nwn, nz, nff)
                                improved = True

    _, d2, wn2, zeta2, ff2 = best2
    yp2, yv2 = sim_second_order_pva(u, u_vel, dt, d2, wn2, zeta2, ff2)
    rmse2, r2_2 = mse_and_r2(y, yp2, skip)
    max_vel, max_acc, overshoot_pct = estimate_kinematic_limits(u[skip:], y[skip:], dt)

    r2d = 180.0 / math.pi
    print(f"\n=== {name} 端到端辨识结果 (N={len(sub)} 原始帧, dt_avg={(ts[-1]-ts[0])/max(1,len(ts)-1)*1000:.1f} ms) ===")
    print(f"  [一阶+延迟] tau_d = {d1*dt*1000:.1f} ms, T = {tau1*1000:.1f} ms (bw={1/(2*math.pi*tau1):.2f} Hz) | RMSE = {rmse1*r2d:.3f} deg, R^2 = {r2_1:.5f}")
    ff_str = f", ff_ratio = {ff2:.3f} (kv={ff2*2*zeta2*wn2:.2f})" if has_uv else ""
    print(f"  [二阶+延迟] tau_d = {d2*dt*1000:.1f} ms, wn = {wn2:.2f} rad/s, zeta = {zeta2:.3f}{ff_str} | 角度 RMSE = {rmse2*r2d:.3f} deg, R^2 = {r2_2:.5f}")
    if has_vy:
        rmse_v, r2_v = mse_and_r2(y_vel_meas, yv2, skip)
        print(f"  [角速度拟合] 实测角速度 RMSE = {rmse_v:.3f} rad/s ({rmse_v*r2d:.2f} deg/s), R^2 = {r2_v:.5f}")
    print(f"  [惯量/动力学极限] 峰值角速度 = {max_vel:.2f} rad/s, 有效峰值角加速度 = {max_acc:.1f} rad/s^2, 峰值超调 = {overshoot_pct:.1f}%")

    # 频率响应诊断（使用二阶含前馈零点模型）
    kv2 = ff2 * 2.0 * zeta2 * wn2
    print("  端到端频率响应（幅值增益 / 总相位滞后）：")
    for freq in [1.0, 2.0, 4.7, 6.0]:
        w = 2.0 * math.pi * freq
        s = 1j * w
        H = (kv2 * s + wn2 * wn2) / (s * s + 2.0 * zeta2 * wn2 * s + wn2 * wn2)
        gain_db = 20.0 * math.log10(max(1e-9, abs(H)))
        phase_deg = math.degrees(cmath.phase(H)) - freq * 360.0 * (d2 * dt)
        equiv_lag_ms = -phase_deg / (360.0 * freq) * 1000.0
        print(f"    f = {freq:3.1f} Hz: gain = {gain_db:+5.2f} dB, phase = {phase_deg:+6.1f} deg (等效总滞后 {equiv_lag_ms:.1f} ms)")

    return {
        "name": name,
        "tau_d1": d1 * dt, "T1": tau1, "rmse1": rmse1, "r2_1": r2_1,
        "tau_d2": d2 * dt, "wn": wn2, "zeta": zeta2, "ff_ratio": ff2, "rmse2": rmse2, "r2_2": r2_2,
        "max_vel": max_vel, "max_acc": max_acc, "overshoot_pct": overshoot_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="/tmp/opencode/gimbal_sysid.csv")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    print(f"读取 {args.csv}，共 {len(rows)} 帧，时长 {rows[-1]['t'] - rows[0]['t']:.2f} s")

    ry = fit_axis(
        "YAW", rows, "uy", "yy",
        {"hold", "yaw_step_pos", "yaw_step_neg", "yaw_step_zero", "yaw_chirp"},
        uv_key="uvy", vy_key="vy",
    )
    rp = fit_axis(
        "PITCH", rows, "up", "yp",
        {"pitch_step_up", "pitch_step_down", "pitch_step_zero", "pitch_chirp"},
        uv_key="uvp", vy_key="vp",
    )

    bs_vals = [r["bs"] for r in rows if r["bs"] is not None and 5.0 < r["bs"] < 35.0]
    if bs_vals:
        mean_bs = sum(bs_vals) / len(bs_vals)
        print(f"\n[弹速统计] 有效样本 {len(bs_vals)} 帧，均值 = {mean_bs:.2f} m/s")

    if ry and rp:
        rec_delay = max(ry["tau_d2"], rp["tau_d2"]) + 0.010  # 纯延迟 + 100Hz 规划周期保持
        rec_yaw_acc = max(15.0, min(120.0, round(ry["max_acc"] * 0.8, 1)))
        rec_pitch_acc = max(20.0, min(150.0, round(rp["max_acc"] * 0.8, 1)))
        print("\n=== 建议写入机器人 YAML（如 configs/standard3.yaml）的 MPC 参数 ===")
        print(f"high_speed_delay_time: {rec_delay:.3f}  # s (纯延迟 {ry['tau_d2']*1000:.0f}ms + 10ms 规划周期)")
        print(f"low_speed_delay_time: {rec_delay:.3f}   # s")
        print(f"air_k: 0.02043                # 实车 17mm 空气阻力系数 (仿真无阻力时用 0.0)")
        print(f"max_yaw_acc: {rec_yaw_acc:.1f}            # rad/s^2 (实测 80% 安全裕度)")
        print(f"max_pitch_acc: {rec_pitch_acc:.1f}          # rad/s^2 (实测 80% 安全裕度)")
        print(f"gimbal_yaw_wn: {ry['wn']:.2f}          # rad/s (R^2={ry['r2_2']:.4f})")
        print(f"gimbal_yaw_zeta: {ry['zeta']:.3f}")
        print(f"gimbal_pitch_wn: {rp['wn']:.2f}        # rad/s (R^2={rp['r2_2']:.4f})")
        print(f"gimbal_pitch_zeta: {rp['zeta']:.3f}")


if __name__ == "__main__":
    main()
