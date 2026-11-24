#!/usr/bin/env python3
"""
视觉 — 电控串口通信稳定性与协议核对脚本（纯 Python 标准库，无需 pyserial）。

协议依据：docs/gimbal_serial_protocol.md / io/gimbal/gimbal.hpp / tools/crc.cpp
  - RX (电控 -> 视觉, 43B): <2sB4ffffffHH  (head='SP', mode, q[4], yaw, yaw_vel, pitch, pitch_vel, bullet_speed, bullet_count, crc16)
  - TX (视觉 -> 电控, 29B): <2sBffffffH    (head='SP', mode, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc, crc16)

用法示例：
  # 1. 只收不发，测试 30 秒上行稳定性与方向/四元数健康度
  python3 tools/test_gimbal_serial.py --port /dev/gimbal --duration 30

  # 2. 双向全双工压测（100 Hz 发送安全 mode=0 帧，不驱动云台）
  python3 tools/test_gimbal_serial.py --port /dev/gimbal --tx-hz 100 --duration 60

  # 3. 导出逐帧 CSV 日志供离线分析
  python3 tools/test_gimbal_serial.py --port /dev/gimbal --duration 20 --csv serial_log.csv

  # 4. 离线自测（虚拟串口 PTY，验证 CRC16 / 帧同步 / 统计逻辑）
  python3 tools/test_gimbal_serial.py --self-test
"""

import argparse
import csv
import math
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
from typing import List, Optional, Tuple

RX_FMT = "<2sB4ffffffHH"
RX_SIZE = struct.calcsize(RX_FMT)  # 43 bytes
TX_FMT = "<2sBffffffH"
TX_SIZE = struct.calcsize(TX_FMT)  # 29 bytes

assert RX_SIZE == 43, f"Unexpected RX_SIZE={RX_SIZE}"
assert TX_SIZE == 29, f"Unexpected TX_SIZE={TX_SIZE}"

MODE_NAMES = {0: "IDLE", 1: "AUTO_AIM", 2: "SMALL_BUFF", 3: "BIG_BUFF"}

# 与 tools/crc.cpp 完全一致的 CRC16 表（初值 0xFFFF，反射多项式 0x8408）
def _build_crc16_table() -> List[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
        table.append(crc & 0xFFFF)
    return table


CRC16_TABLE = _build_crc16_table()


def get_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = ((crc >> 8) ^ CRC16_TABLE[(crc ^ b) & 0xFF]) & 0xFFFF
    return crc


def pack_vision_to_gimbal(
    mode: int = 0,
    yaw: float = 0.0,
    yaw_vel: float = 0.0,
    yaw_acc: float = 0.0,
    pitch: float = 0.0,
    pitch_vel: float = 0.0,
    pitch_acc: float = 0.0,
) -> bytes:
    body = struct.pack(
        "<2sBffffff", b"SP", mode, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc
    )
    return body + struct.pack("<H", get_crc16(body))


def quat_wxyz_to_zyx_ypr(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """返回 ZYX 欧拉角 (yaw, pitch_up_pos, roll)，弧度；pitch 已换算为抬头为正（与电控协议一致）。"""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    # 右手系 ENU 绕 +Y 旋转为低头正、抬头负，取负后转为抬头正
    pitch = -math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return yaw, pitch, roll


def wrap_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def open_serial_raw(port: str, baudrate: int) -> int:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    # 原始 8N1 无流控模式
    attrs[0] = 0  # iflag
    attrs[1] = 0  # oflag
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag
    attrs[3] = 0  # lflag
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0

    baud_map = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
        230400: termios.B230400,
        460800: termios.B460800,
        921600: termios.B921600,
        1000000: getattr(termios, "B1000000", termios.B115200),
        2000000: getattr(termios, "B2000000", termios.B115200),
    }
    b = baud_map.get(baudrate, termios.B115200)
    attrs[4] = b  # ispeed
    attrs[5] = b  # ospeed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * p)))
    return sorted_vals[idx]


def run_monitor(
    fd: int,
    duration: float,
    tx_hz: float,
    tx_mode: int,
    stall_ms: float,
    csv_path: Optional[str],
) -> int:
    stop = False

    def _sig_handler(_sig, _frame):
        nonlocal stop
        stop = True

    old_sigint = signal.signal(signal.SIGINT, _sig_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

    csv_file = open(csv_path, "w", newline="") if csv_path else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(
            [
                "t_rel_s",
                "dt_ms",
                "mode",
                "qw",
                "qx",
                "qy",
                "qz",
                "q_norm",
                "yaw_deg",
                "yaw_vel",
                "pitch_deg",
                "pitch_vel",
                "q_yaw_deg",
                "q_pitch_deg",
                "bullet_speed",
                "bullet_count",
            ]
        )

    buf = bytearray()
    t_start = time.monotonic()
    t_last_report = t_start
    t_last_rx: Optional[float] = None
    t_next_tx = t_start

    total_frames = 0
    win_frames = 0
    crc_errors = 0
    dropped_bytes = 0
    nan_frames = 0
    stall_events = 0
    tx_frames = 0
    tx_errors = 0

    dts_ms: List[float] = []
    max_q_norm_err = 0.0
    max_yaw_diff_deg = 0.0
    max_pitch_diff_deg = 0.0
    pitch_sign_mismatch_cnt = 0
    pitch_out_of_range_cnt = 0
    last_bullet_count: Optional[int] = None
    bullet_inc_total = 0

    last_sample = None

    try:
        while not stop:
            now = time.monotonic()
            elapsed = now - t_start
            if duration > 0 and elapsed >= duration:
                break

            # 1. 可选周期发送（全双工压测）
            if tx_hz > 0 and now >= t_next_tx:
                pkt = pack_vision_to_gimbal(mode=tx_mode)
                try:
                    n = os.write(fd, pkt)
                    if n == len(pkt):
                        tx_frames += 1
                    else:
                        tx_errors += 1
                except OSError:
                    tx_errors += 1
                t_next_tx += 1.0 / tx_hz
                if t_next_tx < now - 0.1:
                    t_next_tx = now + 1.0 / tx_hz

            # 2. 非阻塞等待串口数据
            wait_s = 0.01
            if tx_hz > 0:
                wait_s = max(0.001, min(0.01, t_next_tx - time.monotonic()))
            r, _, _ = select.select([fd], [], [], wait_s)
            if fd in r:
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    chunk = b""
                except OSError as e:
                    print(f"\n[ERROR] 串口读取异常: {e}", file=sys.stderr)
                    break
                if chunk:
                    rx_now = time.monotonic()
                    buf.extend(chunk)

                    # 帧同步与解析
                    while len(buf) >= RX_SIZE:
                        pos = buf.find(b"SP")
                        if pos < 0:
                            dropped_bytes += max(0, len(buf) - 1)
                            del buf[:-1]
                            break
                        if pos > 0:
                            dropped_bytes += pos
                            del buf[:pos]
                        if len(buf) < RX_SIZE:
                            break

                        frame = bytes(buf[:RX_SIZE])
                        crc_calc = get_crc16(frame[:-2])
                        crc_recv = struct.unpack_from("<H", frame, RX_SIZE - 2)[0]
                        if crc_calc != crc_recv:
                            crc_errors += 1
                            dropped_bytes += 1
                            del buf[:1]
                            continue

                        del buf[:RX_SIZE]
                        (
                            _head,
                            mode,
                            qw,
                            qx,
                            qy,
                            qz,
                            yaw,
                            yaw_vel,
                            pitch,
                            pitch_vel,
                            bullet_speed,
                            bullet_count,
                            _crc,
                        ) = struct.unpack(RX_FMT, frame)

                        dt_ms = 0.0
                        if t_last_rx is not None:
                            dt_ms = (rx_now - t_last_rx) * 1000.0
                            dts_ms.append(dt_ms)
                            if dt_ms > stall_ms:
                                stall_events += 1
                        t_last_rx = rx_now
                        total_frames += 1
                        win_frames += 1

                        vals = (qw, qx, qy, qz, yaw, yaw_vel, pitch, pitch_vel, bullet_speed)
                        if any(not math.isfinite(v) for v in vals):
                            nan_frames += 1
                            continue

                        q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                        max_q_norm_err = max(max_q_norm_err, abs(q_norm - 1.0))

                        q_yaw, q_pitch, _q_roll = quat_wxyz_to_zyx_ypr(qw, qx, qy, qz)
                        dy_deg = abs(math.degrees(wrap_rad(yaw - q_yaw)))
                        dp_deg = abs(math.degrees(pitch - q_pitch))
                        max_yaw_diff_deg = max(max_yaw_diff_deg, dy_deg)
                        max_pitch_diff_deg = max(max_pitch_diff_deg, dp_deg)
                        # 当抬头/低头超过 5° 时检查 pitch 与四元数抬头正方向是否反号
                        if abs(math.degrees(q_pitch)) > 5.0 and (pitch * q_pitch < 0.0):
                            pitch_sign_mismatch_cnt += 1
                        p_deg = math.degrees(pitch)
                        if p_deg < -22.0 or p_deg > 32.0:
                            pitch_out_of_range_cnt += 1

                        if last_bullet_count is not None:
                            diff = (bullet_count - last_bullet_count) & 0xFFFF
                            if diff < 100:
                                bullet_inc_total += diff
                        last_bullet_count = bullet_count

                        last_sample = (
                            mode,
                            q_norm,
                            math.degrees(yaw),
                            yaw_vel,
                            math.degrees(pitch),
                            pitch_vel,
                            math.degrees(q_yaw),
                            math.degrees(q_pitch),
                            bullet_speed,
                            bullet_count,
                        )

                        if csv_writer:
                            csv_writer.writerow(
                                [
                                    f"{rx_now - t_start:.4f}",
                                    f"{dt_ms:.3f}",
                                    mode,
                                    f"{qw:.6f}",
                                    f"{qx:.6f}",
                                    f"{qy:.6f}",
                                    f"{qz:.6f}",
                                    f"{q_norm:.6f}",
                                    f"{math.degrees(yaw):.3f}",
                                    f"{yaw_vel:.4f}",
                                    f"{math.degrees(pitch):.3f}",
                                    f"{pitch_vel:.4f}",
                                    f"{math.degrees(q_yaw):.3f}",
                                    f"{math.degrees(q_pitch):.3f}",
                                    f"{bullet_speed:.3f}",
                                    bullet_count,
                                ]
                            )

            # 3. 每秒打印状态摘要
            now = time.monotonic()
            if now - t_last_report >= 1.0:
                dt_win = now - t_last_report
                rx_hz = win_frames / dt_win
                win_frames = 0
                t_last_report = now
                if last_sample:
                    (
                        mode,
                        q_norm,
                        yaw_d,
                        yv,
                        pit_d,
                        pv,
                        qy_d,
                        qp_d,
                        bs,
                        bc,
                    ) = last_sample
                    m_str = MODE_NAMES.get(mode, f"UNK({mode})")
                    print(
                        f"[{now - t_start:6.1f}s] RX {rx_hz:6.1f} Hz | total={total_frames} "
                        f"crc_err={crc_errors} dropB={dropped_bytes} stall={stall_events} | "
                        f"mode={m_str} | Y={yaw_d:+6.1f}°(q:{qy_d:+6.1f}°) P={pit_d:+6.1f}°(q:{qp_d:+6.1f}°) | "
                        f"|q|={q_norm:.4f} v={bs:.1f} cnt={bc}"
                    )
                else:
                    print(
                        f"[{now - t_start:6.1f}s] 等待有效帧... (crc_err={crc_errors}, dropB={dropped_bytes}, tx={tx_frames})"
                    )
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if csv_file:
            csv_file.close()

    # 汇总报告
    total_time = max(1e-6, time.monotonic() - t_start)
    avg_hz = total_frames / total_time
    dts_sorted = sorted(dts_ms)
    err_rate = crc_errors / max(1, total_frames + crc_errors)

    print("\n================ 串口稳定性测试报告 ================")
    print(f"测试时长     : {total_time:.2f} s")
    print(f"接收总帧数   : {total_frames} 帧  (平均 {avg_hz:.1f} Hz)")
    if tx_hz > 0:
        print(f"发送总帧数   : {tx_frames} 帧  (目标 {tx_hz:.1f} Hz, 写失败 {tx_errors})")
    print(f"CRC16 错误   : {crc_errors} 帧  (误码率 {err_rate * 100:.4f}%)")
    print(f"丢弃/失步字节: {dropped_bytes} B")
    print(f"NaN/Inf 帧   : {nan_frames}")
    print(f"断流次数     : {stall_events} (阈值 > {stall_ms:.1f} ms)")
    if dts_sorted:
        avg_dt = sum(dts_sorted) / len(dts_sorted)
        print(
            f"帧间隔 dt    : min={dts_sorted[0]:.2f} ms | avg={avg_dt:.2f} ms | "
            f"p99={percentile(dts_sorted, 0.99):.2f} ms | max={dts_sorted[-1]:.2f} ms"
        )
    print(f"四元数最大偏差: ||q|-1|_max = {max_q_norm_err:.6f}")
    print(f"欧拉角一致性 : |yaw - q_yaw|_max = {max_yaw_diff_deg:.2f}°, |pitch - q_pitch|_max = {max_pitch_diff_deg:.2f}°")
    print(f"发弹计数增量 : +{bullet_inc_total}")

    issues = []
    if total_frames == 0:
        issues.append("未收到任何合法帧（检查串口号、帧头 'SP'、结构体字节数 43B 与 CRC16 算法）")
    if err_rate > 0.001:
        issues.append(f"CRC 误码率偏高 ({err_rate * 100:.3f}% > 0.1%)")
    if nan_frames > 0:
        issues.append(f"出现 {nan_frames} 帧 NaN/Inf")
    if max_q_norm_err > 0.02:
        issues.append(f"四元数未归一化 (最大偏差 {max_q_norm_err:.4f})")
    if pitch_sign_mismatch_cnt > 5:
        issues.append(f"检测到 {pitch_sign_mismatch_cnt} 帧 pitch 与四元数反号（协议要求抬头为正、低头为负）")
    if pitch_out_of_range_cnt > 0:
        issues.append(f"检测到 {pitch_out_of_range_cnt} 帧 pitch 超出 [-20°, +30°] 机械范围")
    if total_frames > 0 and avg_hz < 200.0:
        issues.append(f"上行频率偏低 ({avg_hz:.1f} Hz < 200 Hz，推荐 500~1000 Hz)")
    if stall_events > 0:
        issues.append(f"发生 {stall_events} 次超过 {stall_ms:.1f} ms 的断流/卡顿")

    if not issues:
        print("结论         : PASS（串口通信稳定，协议与数据健康）")
        return 0
    print("结论         : WARN / FAIL")
    for it in issues:
        print(f"  - {it}")
    return 1


def run_self_test() -> int:
    """使用 PTY 虚拟串口模拟下位机 500 Hz 发包并注入噪声，离线验证脚本正确性。"""
    master_fd, slave_fd = pty.openpty()
    # 配置 slave 为 raw 模式
    attrs = termios.tcgetattr(slave_fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    os.set_blocking(master_fd, False)
    os.set_blocking(slave_fd, False)

    stop_sim = threading.Event()
    rx_from_vision = [0]

    def sim_mcu():
        t0 = time.monotonic()
        seq = 0
        tx_buf = bytearray()
        while not stop_sim.is_set():
            t = time.monotonic() - t0
            yaw = 0.4 * math.sin(2.0 * t)
            pitch_up = 0.15 * math.cos(1.5 * t)  # 抬头为正（范围 -20° ~ +30°）
            pitch_enu = -pitch_up                # 右手系 ENU 绕 +Y 旋转角
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            cp, sp = math.cos(pitch_enu * 0.5), math.sin(pitch_enu * 0.5)
            # q = Rz(yaw) * Ry(pitch_enu)
            qw, qx, qy, qz = cy * cp, -sy * sp, cy * sp, sy * cp
            body = struct.pack(
                "<2sB4ffffffH",
                b"SP",
                1,
                qw,
                qx,
                qy,
                qz,
                yaw,
                0.8 * math.cos(2.0 * t),
                pitch_up,
                -0.225 * math.sin(1.5 * t),
                24.5,
                seq // 50,
            )
            pkt = body + struct.pack("<H", get_crc16(body))
            try:
                os.write(master_fd, pkt)
                # 读取上位机发来的 VisionToGimbal 帧
                r, _, _ = select.select([master_fd], [], [], 0)
                if master_fd in r:
                    tx_buf.extend(os.read(master_fd, 4096))
                    while len(tx_buf) >= TX_SIZE:
                        pos = tx_buf.find(b"SP")
                        if pos < 0:
                            del tx_buf[:-1]
                            break
                        if pos > 0:
                            del tx_buf[:pos]
                        if len(tx_buf) < TX_SIZE:
                            break
                        f = bytes(tx_buf[:TX_SIZE])
                        if get_crc16(f[:-2]) == struct.unpack_from("<H", f, TX_SIZE - 2)[0]:
                            rx_from_vision[0] += 1
                            del tx_buf[:TX_SIZE]
                        else:
                            del tx_buf[:1]
            except OSError:
                pass
            seq += 1
            time.sleep(0.002)  # ~500 Hz

    th = threading.Thread(target=sim_mcu, daemon=True)
    th.start()
    try:
        rc = run_monitor(slave_fd, duration=2.2, tx_hz=100.0, tx_mode=0, stall_ms=25.0, csv_path=None)
    finally:
        stop_sim.set()
        th.join(timeout=1.0)
        os.close(master_fd)
        os.close(slave_fd)

    print(f"[Self-Test] 模拟电控收到合法 VisionToGimbal 帧数: {rx_from_vision[0]}")
    if rc == 0 and rx_from_vision[0] >= 150:
        print("[Self-Test] PASS")
        return 0
    print("[Self-Test] FAIL")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉-电控串口通信稳定性与协议核对工具")
    ap.add_argument("--port", default="/dev/gimbal", help="串口设备路径（默认 /dev/gimbal）")
    ap.add_argument("--baud", type=int, default=115200, help="波特率（USB-CDC 忽略，默认 115200）")
    ap.add_argument("--duration", type=float, default=30.0, help="测试时长（秒，0 表示一直跑到 Ctrl+C）")
    ap.add_argument("--tx-hz", type=float, default=0.0, help="下行发送频率 Hz（0=只收不发；建议压测用 100）")
    ap.add_argument("--tx-mode", type=int, default=0, choices=[0, 1, 2], help="下行发送 mode（默认 0 安全空闲）")
    ap.add_argument("--stall-ms", type=float, default=20.0, help="判定断流/卡顿的帧间隔阈值 (ms)")
    ap.add_argument("--csv", default="", help="可选：保存逐帧日志到 CSV 文件")
    ap.add_argument("--self-test", action="store_true", help="使用虚拟串口 PTY 离线自测")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    try:
        fd = open_serial_raw(args.port, args.baud)
    except OSError as e:
        print(f"[ERROR] 打开串口 {args.port} 失败: {e}", file=sys.stderr)
        return 2

    print(
        f"打开 {args.port} 成功 | duration={args.duration}s | tx_hz={args.tx_hz} (mode={args.tx_mode}) | stall>{args.stall_ms}ms"
    )
    try:
        return run_monitor(fd, args.duration, args.tx_hz, args.tx_mode, args.stall_ms, args.csv or None)
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
