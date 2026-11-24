#!/usr/bin/env python3
"""
仿真云台 PID (kp, kd) 离线代价函数优化脚本（纯 Python 3 标准库）。
精确复现 bevy_robomaster_simulator/src/systems/gimbal_pid.rs 的离散运动学闭环：
  rate_k = clip(kp * e_k + kd * (e_k - e_{k-1}) / dt_k, -max_rate, max_rate)
  y_{k+1} = y_k + rate_k * dt_k
结合 daedalus 实测帧时间波动 dt_k，在阶跃 + 4.7 Hz 小陀螺锯齿波 + 最坏掉帧工况下
最小化综合代价函数 J(kp, kd)，输出最优 (kp, kd) 与对比表。
"""

import argparse
import math
import re
import statistics


def parse_dt_from_log(log_path):
    """从 daedalus 日志中提取真实 frame_time (s) 序列。"""
    pat = re.compile(r"frame_time\s*:\s*([0-9.]+)ms")
    dts = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    ms = float(m.group(1))
                    if 2.0 <= ms <= 80.0:
                        dts.append(ms * 1e-3)
    except OSError:
        pass
    return dts


def build_dt_sequence(raw_dts, n_steps):
    """基于实测帧时间构造周期平滑抖动序列（保留真实均值、标准差与 P95/最大值）。"""
    if not raw_dts:
        raw_dts = [0.010, 0.011, 0.012, 0.014, 0.017, 0.019, 0.013, 0.009]
    seq = []
    m = len(raw_dts)
    for i in range(n_steps):
        seq.append(raw_dts[i % m])
    return seq


def sim_pid(ref_fn, dt_seq, kp, kd, max_rate=100.0, mpc_lead_s=0.0, mpc_margin=0.026):
    """精确模拟 gimbal_pid.rs（含 1 帧指令-执行延迟，可选叠加 MPC 预补偿前馈）。"""
    y = 0.0
    prev_e = 0.0
    prev_rate = 0.0
    t = 0.0
    r_prev = ref_fn(0.0)

    err_sq = 0.0
    d_rate_sq = 0.0
    ys = []
    rs = []
    n = len(dt_seq)
    for dt in dt_seq:
        t += dt
        r_now = ref_fn(t)
        # 若开启 MPC 前馈：近似 Planner 逆模型超前补偿，并受 ±mpc_margin 动态盒约束
        if mpc_lead_s > 0.0:
            r_future = ref_fn(t + mpc_lead_s)
            delta = max(-mpc_margin, min(mpc_margin, r_future - r_now))
            r_send = r_now + delta
        else:
            r_send = r_now
        # process_subscription 在 Last 更新 target，下一帧 Update 生效 -> 1 帧延迟
        r_cmd = r_prev
        r_prev = r_send

        e = r_cmd - y
        deriv = (e - prev_e) / dt
        prev_e = e
        rate = kp * e + kd * deriv
        if rate > max_rate:
            rate = max_rate
        elif rate < -max_rate:
            rate = -max_rate

        y += rate * dt
        err_sq += (r_now - y) ** 2
        d_rate_sq += (rate - prev_rate) ** 2
        prev_rate = rate
        ys.append(y)
        rs.append(r_now)

    rmse = math.sqrt(err_sq / max(1, n))
    drate_rms = math.sqrt(d_rate_sq / max(1, n))
    return rmse, drate_rms, ys, rs


def eval_cost(kp, kd, dt_seq, dt_p95, dt_max, with_mpc=True):
    """
    综合代价函数 J(kp, kd)：
      1) 小陀螺 4.7 Hz 锯齿波跟踪 RMSE（主指标，with_mpc=True 时叠加 MPC 前馈预补偿）
      2) 10° 阶跃响应 RMSE + 过冲平方惩罚
      3) 角速度指令抖动 RMS(Δrate) 惩罚
      4) 最坏掉帧 dt_max 下的离散稳定性与过冲惩罚
      5) with_mpc=True 时，额外惩罚 kd 引入的分子零点与 MPC 逆模型失配（双重前馈振铃）
    """
    # MPC 逆模型在 1/kp 滞后上叠加超前量（受 planner.cpp ±0.026 rad 盒约束）
    mpc_lead = min(0.030, 1.0 / max(10.0, kp)) if with_mpc else 0.0

    # 场景 A：10° 阶跃
    step_amp = math.radians(10.0)
    step_rmse, step_drate, step_ys, _ = sim_pid(
        lambda t: step_amp if t >= 0.05 else 0.0, dt_seq[:120], kp, kd, mpc_lead_s=mpc_lead
    )
    step_peak = max(step_ys) if step_ys else 0.0
    step_overshoot = max(0.0, step_peak - step_amp)
    step_overshoot_pct = 100.0 * step_overshoot / step_amp

    # 场景 B：4.7 Hz 小陀螺锯齿波（w=7.42 rad/s, r=0.213m, dist=2.6m -> ±4.69°）
    w = 7.42
    r = 0.213
    dist = 2.6
    half_period = (math.pi / 2.0) / w  # ~0.2117s (4.72 Hz)

    def sawtooth(t):
        phase = ((t % half_period) / half_period - 0.5) * (math.pi / 2.0)
        return math.atan2(r * math.sin(phase), dist - r * math.cos(phase))

    saw_rmse, saw_drate, saw_ys, saw_rs = sim_pid(
        sawtooth, dt_seq, kp, kd, mpc_lead_s=mpc_lead
    )
    saw_bound = max(abs(x) for x in saw_rs)
    saw_overshoot = max(0.0, max(abs(y) for y in saw_ys) - saw_bound)

    # 场景 C：最坏掉帧 dt_max 下阶跃鲁棒性（防止掉帧时振铃/发散）
    dip_seq = [dt_max] * 40
    dip_rmse, _, dip_ys, _ = sim_pid(
        lambda t: step_amp if t >= dt_max else 0.0, dip_seq, kp, kd, mpc_lead_s=mpc_lead
    )
    dip_os = max(0.0, (max(dip_ys) if dip_ys else 0.0) - step_amp)
    # 特征方程 z^2 - (1 - kp*dt - kd)*z - kd = 0 在 dt_max 下的最大模长
    b = -(1.0 - kp * dt_max - kd)
    c = -kd
    disc = b * b - 4.0 * c
    if disc >= 0.0:
        sr = math.sqrt(disc)
        rho_max = max(abs((-b + sr) * 0.5), abs((-b - sr) * 0.5))
    else:
        rho_max = math.sqrt(max(0.0, c))
    stab_penalty = 0.0
    if rho_max >= 0.90:
        stab_penalty += 10.0 * (rho_max - 0.90) ** 2
    if rho_max >= 0.99:
        stab_penalty += 100.0
    # 若开启 MPC，底层 PID 的 kd 会引入非最小相位/分子零点，导致与二阶低通 MPC 失配及双重前馈
    mismatch_penalty = (0.04 * (kd ** 2)) if with_mpc else 0.0

    # 综合代价（角度单位 rad，权重对齐实际命中率与平滑度）
    cost = (
        1.0 * (saw_rmse ** 2)
        + 0.35 * (step_rmse ** 2)
        + 6.0 * (step_overshoot ** 2)
        + 8.0 * (saw_overshoot ** 2)
        + 4.0 * (dip_os ** 2)
        + 1.5e-6 * (saw_drate ** 2)
        + stab_penalty
        + mismatch_penalty
    )
    return {
        "cost": cost,
        "saw_rmse_deg": math.degrees(saw_rmse),
        "step_rmse_deg": math.degrees(step_rmse),
        "step_os_pct": step_overshoot_pct,
        "saw_os_deg": math.degrees(saw_overshoot),
        "dip_os_pct": 100.0 * dip_os / step_amp,
        "saw_drate": saw_drate,
        "rho_max": rho_max,
    }


def main():
    parser = argparse.ArgumentParser(description="仿真云台 PID (kp, kd) 代价函数优化")
    parser.add_argument("--log", default="/tmp/opencode/daedalus_live.log", help="daedalus 日志路径")
    parser.add_argument("--max-dt-ms", type=float, default=None, help="可选：手动覆盖最坏帧时间 (ms)")
    args = parser.parse_args()

    raw_dts = parse_dt_from_log(args.log)
    if raw_dts:
        # 取最近 60 个样本反映当前高帧率状态
        recent = raw_dts[-60:]
        s_dts = sorted(recent)
        dt_mean = statistics.mean(recent)
        dt_p95 = s_dts[int(0.95 * (len(s_dts) - 1))]
        dt_max = s_dts[-1]
        print(
            f"[DT] 从 {args.log} 读取 {len(recent)} 个帧时间样本: "
            f"mean={dt_mean*1e3:.2f}ms ({1.0/dt_mean:.1f} FPS), "
            f"p95={dt_p95*1e3:.2f}ms, max={dt_max*1e3:.2f}ms ({1.0/dt_max:.1f} FPS)"
        )
    else:
        recent = []
        dt_mean, dt_p95, dt_max = 0.0115, 0.019, 0.020
        print("[DT] 未读到日志，使用默认分布: mean=11.5ms, p95=19.0ms, max=20.0ms")

    if args.max_dt_ms is not None:
        dt_max = args.max_dt_ms * 1e-3

    dt_seq = build_dt_sequence(recent, 600)

    # 粗网格搜索
    best = None
    best_pair = None
    for kp_i in range(20, 121, 2):
        kp = float(kp_i)
        for kd_i in range(0, 31):
            kd = kd_i * 0.01
            res = eval_cost(kp, kd, dt_seq, dt_p95, dt_max)
            if best is None or res["cost"] < best["cost"]:
                best = res
                best_pair = (kp, kd)

    # 局部细化
    kp0, kd0 = best_pair
    for dkp in [i * 0.25 for i in range(-12, 13)]:
        kp = max(5.0, kp0 + dkp)
        for dkd in [j * 0.002 for j in range(-10, 11)]:
            kd = max(0.0, kd0 + dkd)
            res = eval_cost(kp, kd, dt_seq, dt_p95, dt_max)
            if res["cost"] < best["cost"]:
                best = res
                best_pair = (kp, kd)

    print("\n=== PID 参数对比 ===")
    hdr = f"{'配置':<18} {'kp':>6} {'kd':>6} {'J(×1e4)':>9} {'锯齿RMSE':>9} {'阶跃OS%':>8} {'掉帧OS%':>8} {'Δrate':>7} {'ρ_max':>6}"
    print(hdr)
    print("-" * len(hdr))
    candidates = [
        ("当前(原厂)", 50.0, 0.10),
        ("纯P(kp=35)", 35.0, 0.00),
        ("纯P(kp=50)", 50.0, 0.00),
        ("纯P(kp=60)", 60.0, 0.00),
        ("最优(kp*,kd*)", best_pair[0], best_pair[1]),
    ]
    for name, kp, kd in candidates:
        r = eval_cost(kp, kd, dt_seq, dt_p95, dt_max)
        print(
            f"{name:<18} {kp:6.2f} {kd:6.3f} {r['cost']*1e4:9.3f} "
            f"{r['saw_rmse_deg']:8.2f}° {r['step_os_pct']:7.1f}% "
            f"{r['dip_os_pct']:7.1f}% {r['saw_drate']:7.1f} {r['rho_max']:6.3f}"
        )

    print(
        f"\n推荐写入 .sim/run/config.toml:\n"
        f"  kp = {best_pair[0]:.2f}\n"
        f"  ki = 0.0\n"
        f"  kd = {best_pair[1]:.3f}"
    )


if __name__ == "__main__":
    main()
