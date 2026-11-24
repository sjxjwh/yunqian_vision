#!/usr/bin/env python3
"""
从机器人 URDF 机械文件自动提取视觉标定初值、枪口偏移与云台构型参数。

输出内容：
  1. R_camera2gimbal（相机光学系 z前/x右/y下 -> 云台系 x前/y左/z上）与 t_camera2gimbal
  2. muzzle_z_offset 与 t_muzzle2gimbal（枪口相对云台原点的平移，单位 m）
  3. 云台 yaw/pitch 关节构型（轴向、限位、最大速度、力矩与对应 link 惯量）
  4. 可选：直接将外参与枪口参数写回指定 configs/*.yaml

用法示例：
  python3 tools/urdf_to_vision_yaml.py --urdf robot.urdf
  python3 tools/urdf_to_vision_yaml.py --urdf robot.urdf \\
      --gimbal-link pitch_link --camera-link camera_link --muzzle-link muzzle_link \\
      --apply configs/standard3.yaml
"""

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------- 3D 刚体变换工具 -----------------------------

def rpy_to_rot(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """URDF 规范：固定轴 XYZ 欧拉角（等价于内旋 ZYX：Rz(yaw) * Ry(pitch) * Rx(roll)）。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_T(A: List[List[float]]) -> List[List[float]]:
    return [[A[j][i] for j in range(3)] for i in range(3)]


def mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(A[i][j] * v[j] for j in range(3)) for i in range(3)]


def tf_mul(
    T1: Tuple[List[List[float]], List[float]],
    T2: Tuple[List[List[float]], List[float]],
) -> Tuple[List[List[float]], List[float]]:
    R1, t1 = T1
    R2, t2 = T2
    R = mat_mul(R1, R2)
    Rt2 = mat_vec(R1, t2)
    t = [t1[i] + Rt2[i] for i in range(3)]
    return R, t


def tf_inv(T: Tuple[List[List[float]], List[float]]) -> Tuple[List[List[float]], List[float]]:
    R, t = T
    RT = mat_T(R)
    neg_RT_t = [-x for x in mat_vec(RT, t)]
    return RT, neg_RT_t


def tf_identity() -> Tuple[List[List[float]], List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0]


# ROS 相机机身系 (x前, y左, z上) <- OpenCV/ROS 光学系 (x右, y下, z前)
# p_body = R_optical2body * p_optical:
#   x_body = z_opt, y_body = -x_opt, z_body = -y_opt
R_OPTICAL_TO_BODY = [
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
]


def parse_vec3(s: Optional[str], default=(0.0, 0.0, 0.0)) -> List[float]:
    if not s:
        return list(default)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got: {s!r}")
    return vals


# ----------------------------- URDF 解析 -----------------------------

class URDFModel:
    def __init__(self, path: Path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.links: Dict[str, dict] = {}
        self.joints: Dict[str, dict] = {}
        self.adj: Dict[str, List[Tuple[str, Tuple[List[List[float]], List[float]], str]]] = {}

        for lk in self.root.findall("link"):
            name = lk.attrib["name"]
            info = {"name": name, "mass": None, "inertia": None, "com": None}
            inertial = lk.find("inertial")
            if inertial is not None:
                m_el = inertial.find("mass")
                if m_el is not None and "value" in m_el.attrib:
                    info["mass"] = float(m_el.attrib["value"])
                orig = inertial.find("origin")
                if orig is not None:
                    info["com"] = parse_vec3(orig.attrib.get("xyz"))
                inert = inertial.find("inertia")
                if inert is not None:
                    info["inertia"] = {
                        k: float(inert.attrib.get(k, 0.0))
                        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                    }
            self.links[name] = info
            self.adj.setdefault(name, [])

        for jt in self.root.findall("joint"):
            name = jt.attrib["name"]
            jtype = jt.attrib.get("type", "fixed")
            parent = jt.find("parent").attrib["link"]
            child = jt.find("child").attrib["link"]
            orig = jt.find("origin")
            xyz = parse_vec3(orig.attrib.get("xyz") if orig is not None else None)
            rpy = parse_vec3(orig.attrib.get("rpy") if orig is not None else None)
            axis_el = jt.find("axis")
            axis = parse_vec3(axis_el.attrib.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
            lim_el = jt.find("limit")
            limit = None
            if lim_el is not None:
                limit = {
                    k: float(lim_el.attrib[k])
                    for k in ("lower", "upper", "velocity", "effort")
                    if k in lim_el.attrib
                }

            T_p_c = (rpy_to_rot(*rpy), xyz)
            self.joints[name] = {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "limit": limit,
                "T_parent_child": T_p_c,
            }
            self.adj.setdefault(parent, []).append((child, T_p_c, name))
            self.adj.setdefault(child, []).append((parent, tf_inv(T_p_c), name))

    def transform(self, target_frame: str, source_frame: str) -> Tuple[List[List[float]], List[float]]:
        """计算 T_target_source（满足 p_target = R * p_source + t）。"""
        if target_frame not in self.adj:
            raise KeyError(f"Link not found in URDF: {target_frame}")
        if source_frame not in self.adj:
            raise KeyError(f"Link not found in URDF: {source_frame}")

        visited = {target_frame}
        queue: List[Tuple[str, Tuple[List[List[float]], List[float]]]] = [
            (target_frame, tf_identity())
        ]
        while queue:
            cur, T_target_cur = queue.pop(0)
            if cur == source_frame:
                return T_target_cur
            for nxt, T_cur_nxt, _ in self.adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, tf_mul(T_target_cur, T_cur_nxt)))
        raise RuntimeError(f"No kinematic path between {target_frame} and {source_frame}")

    def guess_link(self, patterns: List[str]) -> Optional[str]:
        names = list(self.links.keys())
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            for n in names:
                if rx.search(n):
                    return n
        return None


def fmt_mat9(R: List[List[float]]) -> str:
    flat = [R[i][j] for i in range(3) for j in range(3)]
    return "[" + ", ".join(f"{v:.6f}" for v in flat) + "]"


def fmt_vec3(v: List[float]) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def update_yaml_file(yaml_path: Path, updates: Dict[str, str]) -> None:
    """保留注释与格式，按顶层键原地更新 YAML 字段；缺失的键追加到末尾。"""
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = set()
    new_lines = []
    for line in lines:
        replaced = False
        for key, val_str in updates.items():
            if re.match(rf"^{re.escape(key)}\s*:", line):
                new_lines.append(f"{key}: {val_str}")
                found.add(key)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    missing = [k for k in updates if k not in found]
    if missing:
        new_lines.append("")
        new_lines.append("# 以下外参/枪口几何由 tools/urdf_to_vision_yaml.py 自动生成")
        for k in missing:
            new_lines.append(f"{k}: {updates[k]}")

    yaml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="从 URDF 机械文件提取视觉外参、枪口坐标与云台构型")
    ap.add_argument("--urdf", required=True, help="URDF 文件路径")
    ap.add_argument("--gimbal-link", default="", help="云台参考 link（默认自动匹配 pitch/gimbal）")
    ap.add_argument("--camera-link", default="", help="相机 link（默认自动匹配 camera_optical/camera）")
    ap.add_argument("--muzzle-link", default="", help="枪口/发射口 link（默认自动匹配 muzzle/shooter/barrel）")
    ap.add_argument(
        "--camera-is-optical",
        action="store_true",
        help="声明 --camera-link 已是 OpenCV/ROS 光学系 (x右/y下/z前)；否则按机身系 (x前/y左/z上) 自动转光学系",
    )
    ap.add_argument("--apply", default="", help="可选：将提取结果直接写入指定 YAML 配置文件")
    args = ap.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"[ERROR] URDF 文件不存在: {urdf_path}", file=sys.stderr)
        return 1

    model = URDFModel(urdf_path)
    gimbal_link = args.gimbal_link or model.guess_link(
        [r"gimbal.*pitch", r"pitch.*link", r"gimbal_link", r"gimbal"]
    )
    camera_link = args.camera_link or model.guess_link(
        [r"camera.*optical", r"cam.*optical", r"camera_link", r"camera"]
    )
    muzzle_link = args.muzzle_link or model.guess_link(
        [r"muzzle", r"shooter", r"barrel", r"gun"]
    )

    print(f"=== URDF 解析: {urdf_path} ===")
    print(f"共发现 {len(model.links)} 个 link, {len(model.joints)} 个 joint")
    print(f"选定 link: gimbal={gimbal_link}, camera={camera_link}, muzzle={muzzle_link}")

    # 云台活动关节构型摘要
    print("\n--- 云台/运动关节构型 ---")
    for jt in model.joints.values():
        if jt["type"] == "fixed":
            continue
        lim = jt["limit"] or {}
        lo = f"{math.degrees(lim['lower']):.1f}°" if "lower" in lim else "-∞"
        hi = f"{math.degrees(lim['upper']):.1f}°" if "upper" in lim else "+∞"
        vel = f"{lim['velocity']:.2f} rad/s" if "velocity" in lim else "N/A"
        eff = f"{lim['effort']:.2f} N·m" if "effort" in lim else "N/A"
        child_info = model.links.get(jt["child"], {})
        mass = f"{child_info['mass']:.3f} kg" if child_info.get("mass") is not None else "N/A"
        inert = child_info.get("inertia")
        inert_str = (
            f"Ixx={inert['ixx']:.5f}, Iyy={inert['iyy']:.5f}, Izz={inert['izz']:.5f}"
            if inert
            else "N/A"
        )
        print(
            f"  joint {jt['name']} ({jt['type']}): {jt['parent']} -> {jt['child']} | "
            f"axis={fmt_vec3(jt['axis'])} | limit=[{lo}, {hi}] | vmax={vel} | τmax={eff} | "
            f"child_mass={mass}, {inert_str}"
        )

    updates: Dict[str, str] = {}

    if gimbal_link and camera_link:
        R_g_c, t_g_c = model.transform(gimbal_link, camera_link)
        is_opt = args.camera_is_optical or ("optical" in camera_link.lower())
        if not is_opt:
            # R_gimbal_optical = R_gimbal_cambody * R_cambody_optical
            R_g_c = mat_mul(R_g_c, R_OPTICAL_TO_BODY)
        updates["R_camera2gimbal"] = fmt_mat9(R_g_c)
        updates["t_camera2gimbal"] = fmt_vec3(t_g_c)
        print(f"\n--- 相机 -> 云台外参 ({'光学系' if is_opt else '机身系自动转光学系'}) ---")
        print(f"R_camera2gimbal: {updates['R_camera2gimbal']}")
        print(f"t_camera2gimbal: {updates['t_camera2gimbal']}")
    else:
        print("\n[WARN] 未匹配到 gimbal 或 camera link，请用 --gimbal-link / --camera-link 手动指定")

    if gimbal_link and muzzle_link:
        _, t_g_m = model.transform(gimbal_link, muzzle_link)
        updates["muzzle_z_offset"] = f"{t_g_m[2]:.6f}"
        updates["t_muzzle2gimbal"] = fmt_vec3(t_g_m)
        print("\n--- 枪口相对云台坐标 ---")
        print(f"t_muzzle2gimbal: {updates['t_muzzle2gimbal']}")
        print(f"muzzle_z_offset: {updates['muzzle_z_offset']}")
    else:
        print("\n[WARN] 未匹配到 muzzle link，如需导出枪口坐标请用 --muzzle-link 指定")

    if args.apply:
        if not updates:
            print("[ERROR] 没有可写入的参数", file=sys.stderr)
            return 1
        yaml_path = Path(args.apply)
        update_yaml_file(yaml_path, updates)
        print(f"\n[OK] 已将 {list(updates.keys())} 写入 {yaml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
