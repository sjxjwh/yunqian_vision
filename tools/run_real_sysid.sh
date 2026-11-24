#!/usr/bin/env bash
# 实车云台传递函数与惯量极限一键辨识脚本（NUC / Orin NX 通用）。
# 用法示例：
#   ./tools/run_real_sysid.sh configs/standard3.yaml --bus gimbal --yaw-min -180 --yaw-max 180 --pitch-min -18 --pitch-max 22
#   ./tools/run_real_sysid.sh configs/sentry.yaml --bus cboard --yaw-min -40 --yaw-max 40 --pitch-min -15 --pitch-max 20

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/real_gimbal_sysid"
FIT_PY="${ROOT_DIR}/tools/fit_gimbal_tf.py"

CONFIG="${1:-configs/standard3.yaml}"
shift || true

BUS="gimbal"
YAW_MIN="-30"
YAW_MAX="30"
PITCH_MIN="-15"
PITCH_MAX="15"
MARGIN="2"
YAW_AMP="12"
PITCH_AMP="6"
RATE="100"
F_MAX="8"
OUT="${ROOT_DIR}/gimbal_sysid_real.csv"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bus) BUS="$2"; shift 2 ;;
    --yaw-min) YAW_MIN="$2"; shift 2 ;;
    --yaw-max) YAW_MAX="$2"; shift 2 ;;
    --pitch-min) PITCH_MIN="$2"; shift 2 ;;
    --pitch-max) PITCH_MAX="$2"; shift 2 ;;
    --margin) MARGIN="$2"; shift 2 ;;
    --yaw-amp) YAW_AMP="$2"; shift 2 ;;
    --pitch-amp) PITCH_AMP="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --f-max) F_MAX="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-ff) EXTRA_ARGS+=("-with-ff=false"); shift ;;
    --skip-mode-wait) EXTRA_ARGS+=("-skip-mode-wait=true"); shift ;;
    -h|--help)
      echo "用法: $0 <config.yaml> [--bus gimbal|cboard] [--yaw-min -30] [--yaw-max 30] [--pitch-min -15] [--pitch-max 15] [--margin 2] [--yaw-amp 12] [--pitch-amp 6] [--rate 100] [--f-max 8] [--out file.csv]"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ ! -x "${BIN}" ]]; then
  echo "未找到 ${BIN}，请先编译："
  echo "  cmake --build build --target real_gimbal_sysid -j\$(nproc)"
  exit 1
fi

echo "================ 实车云台参数辨识安全确认 ================"
echo "配置文件 : ${CONFIG}"
echo "通信通道 : ${BUS}"
echo "Yaw 限位 : [${YAW_MIN}°, ${YAW_MAX}°]  (余量 ${MARGIN}°, 期望幅值 ${YAW_AMP}°)"
echo "Pitch限位: [${PITCH_MIN}°, ${PITCH_MAX}°]  (余量 ${MARGIN}°, 期望幅值 ${PITCH_AMP}°)"
echo "控制频率 : ${RATE} Hz | 最高扫频: ${F_MAX} Hz"
echo "输出 CSV : ${OUT}"
echo "安全提示 : 请确保云台周围无障碍物、摩擦轮已关闭；随时可拨出 AUTO_AIM 或 Ctrl+C 急停。"
echo "=========================================================="
read -r -p "确认限位无误并开始辨识? [y/N] " ans
if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
  echo "已取消。"
  exit 0
fi

"${BIN}" "${CONFIG}" \
  -bus="${BUS}" \
  -yaw-min="${YAW_MIN}" \
  -yaw-max="${YAW_MAX}" \
  -pitch-min="${PITCH_MIN}" \
  -pitch-max="${PITCH_MAX}" \
  -margin="${MARGIN}" \
  -yaw-amp="${YAW_AMP}" \
  -pitch-amp="${PITCH_AMP}" \
  -rate="${RATE}" \
  -f-max="${F_MAX}" \
  -out="${OUT}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "开始自动拟合传递函数与惯量动力学参数..."
python3 "${FIT_PY}" "${OUT}"
