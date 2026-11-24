#!/usr/bin/env bash
# 仿真闭环一键启动脚本（daedalus 仿真器 + sim_standard 自瞄）。
# Ctrl+C 会自动同时清理仿真器与视觉进程。
#
# 用法示例：
#   ./scripts/run_sim.sh                      # 同时启动仿真器与自瞄（默认 configs/sim.yaml）
#   ./scripts/run_sim.sh --profile            # 启动并每秒打印分项耗时与帧率抖动
#   ./scripts/run_sim.sh --corrector          # 启用视觉弹道闭环预览
#   ./scripts/run_sim.sh --vision-only        # 仅启动自瞄（仿真器已在运行）
#   ./scripts/run_sim.sh --sim-only           # 仅启动仿真器

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

CONFIG="configs/sim.yaml"
START_SIM=1
START_VISION=1
USE_WAYLAND=0
SIM_LOG="${ROOT_DIR}/.sim/daedalus.log"
VISION_ARGS=()

usage() {
  cat <<EOF
用法: $(basename "$0") [脚本选项] [sim_standard 透传参数...]

脚本选项:
  -c, --config FILE   配置文件（默认: configs/sim.yaml）
  --vision-only       仅启动 sim_standard（不启动 daedalus）
  --sim-only          仅启动 daedalus 仿真器（前台运行）
  --wayland           强制 WINIT_UNIX_BACKEND=wayland
  -h, --help          显示帮助

常用透传给 sim_standard 的参数:
  --profile           每秒打印视觉/控制分项耗时与抖动
  --corrector         启用视觉弹道闭环预览
  --mode=planner      强制 MPC Planner（或 --mode=aimer）
  --log=FILE          导出逐帧评估指标 CSV
  --gimbal-log=FILE   导出云台/指令 CSV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG="$2"; shift 2 ;;
    --vision-only) START_SIM=0; shift ;;
    --sim-only) START_VISION=0; shift ;;
    --wayland) USE_WAYLAND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) VISION_ARGS+=("$1"); shift ;;
  esac
done

DAEDALUS_BIN="${ROOT_DIR}/.rust/target/release/daedalus"
SIM_RUN_DIR="${ROOT_DIR}/.sim/run"

if [[ "${START_SIM}" -eq 1 ]]; then
  if [[ ! -x "${DAEDALUS_BIN}" ]]; then
    echo "[错误] 未找到仿真器可执行文件: ${DAEDALUS_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${SIM_RUN_DIR}" ]]; then
    echo "[错误] 未找到仿真运行目录: ${SIM_RUN_DIR}" >&2
    exit 1
  fi
fi

if [[ "${START_VISION}" -eq 1 ]]; then
  require_binary "sim_standard"
fi

setup_vision_env

SIM_PID=""
cleanup() {
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    echo ""
    echo "[sim] 正在停止 daedalus (PID=${SIM_PID})..."
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

launch_daedalus_env() {
  setup_sim_rust_env
  if [[ "${USE_WAYLAND}" -eq 1 ]]; then
    export WINIT_UNIX_BACKEND=wayland
  fi
}

if [[ "${START_SIM}" -eq 1 && "${START_VISION}" -eq 0 ]]; then
  echo "[sim] 前台启动 daedalus（工作目录: ${SIM_RUN_DIR}）..."
  launch_daedalus_env
  cd "${SIM_RUN_DIR}"
  exec "${DAEDALUS_BIN}"
fi

if [[ "${START_SIM}" -eq 1 ]]; then
  # 避免重复启动多个 daedalus 抢共享内存
  if pgrep -x daedalus >/dev/null 2>&1; then
    echo "[sim] 检测到 daedalus 已在运行，直接复用现有实例。"
  else
    echo "[sim] 后台启动 daedalus（日志: ${SIM_LOG}）..."
    (
      launch_daedalus_env
      cd "${SIM_RUN_DIR}"
      exec "${DAEDALUS_BIN}"
    ) >"${SIM_LOG}" 2>&1 &
    SIM_PID=$!
    sleep 1.5
    if ! kill -0 "${SIM_PID}" 2>/dev/null; then
      echo "[错误] daedalus 启动失败，最后 30 行日志：" >&2
      tail -n 30 "${SIM_LOG}" >&2 || true
      exit 1
    fi
  fi
fi

echo "========================================================"
echo "  仿真闭环启动 | 配置: ${CONFIG}"
echo "  WebViewer   : http://127.0.0.1:8090"
echo "  仿真快捷键   : 按 F5 开启/关闭自瞄跟随，Ctrl+C 退出"
echo "========================================================"

cd "${ROOT_DIR}"
"${BUILD_DIR}/sim_standard" "${VISION_ARGS[@]}" "${CONFIG}"
