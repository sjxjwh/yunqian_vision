#!/usr/bin/env bash
# 实机自瞄启动脚本（standard_mpc，NUC / Orin NX 通用）。
# 默认采用安全调试模式（--no-fire --web），避免误发弹；实弹需显式加 --live。
#
# 用法示例：
#   ./scripts/run_standard.sh                         # 安全调试（configs/standard3.yaml, --no-fire --web）
#   ./scripts/run_standard.sh --live                  # 实弹模式（允许开火，默认关 WebViewer）
#   ./scripts/run_standard.sh --live --web            # 实弹模式 + 开启 WebViewer
#   ./scripts/run_standard.sh -c configs/standard4.yaml --record

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

CONFIG="configs/standard3.yaml"
LIVE_MODE=0
ENABLE_WEB=1
WEB_SET=0
FORCE_AIM=0
ENABLE_RECORD=0
PORT="8090"
EXTRA_ARGS=()

usage() {
  cat <<EOF
用法: $(basename "$0") [选项]

选项:
  -c, --config FILE   配置文件（默认: configs/standard3.yaml）
  --debug             安全调试模式：禁止发弹 + 开启 WebViewer（默认）
  --live              实弹模式：允许发弹（不加 --no-fire），默认关闭 WebViewer
  --web               开启 WebViewer
  --no-web            关闭 WebViewer
  -p, --port PORT     WebViewer 端口（默认: 8090）
  --record            开启视频录制
  --force-aim         忽略电控 mode，强制进入自瞄（仅排障使用，注意安全！）
  -h, --help          显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG="$2"; shift 2 ;;
    --debug) LIVE_MODE=0; shift ;;
    --live) LIVE_MODE=1; shift ;;
    --web) ENABLE_WEB=1; WEB_SET=1; shift ;;
    --no-web) ENABLE_WEB=0; WEB_SET=1; shift ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --record) ENABLE_RECORD=1; shift ;;
    --force-aim) FORCE_AIM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# 实弹模式默认关闭 WebViewer 以节省带宽与 CPU，除非显式指定 --web
if [[ "${LIVE_MODE}" -eq 1 && "${WEB_SET}" -eq 0 ]]; then
  ENABLE_WEB=0
fi

require_binary "standard_mpc"
setup_vision_env

ARGS=()
if [[ "${LIVE_MODE}" -eq 0 ]]; then
  ARGS+=(--no-fire)
fi
if [[ "${ENABLE_WEB}" -eq 1 ]]; then
  ARGS+=(--web "--port=${PORT}")
fi
if [[ "${ENABLE_RECORD}" -eq 1 ]]; then
  ARGS+=(--record)
fi
if [[ "${FORCE_AIM}" -eq 1 ]]; then
  ARGS+=(--force-aim)
fi

LOCAL_IP="$(detect_local_ip)"
echo "========================================================"
echo "  实机自瞄启动 (standard_mpc)"
echo "  配置文件  : ${CONFIG}"
if [[ "${LIVE_MODE}" -eq 1 ]]; then
  echo "  开火状态  : [实弹模式] 允许发弹！"
else
  echo "  开火状态  : [安全调试] 已强制 --no-fire"
fi
if [[ "${FORCE_AIM}" -eq 1 ]]; then
  echo "  模式覆盖  : [警告] 已启用 --force-aim（上电立即跟随）"
fi
if [[ "${ENABLE_WEB}" -eq 1 ]]; then
  echo "  WebViewer : http://${LOCAL_IP}:${PORT}"
fi
echo "========================================================"

cd "${ROOT_DIR}"
exec "${BUILD_DIR}/standard_mpc" "${ARGS[@]}" "${EXTRA_ARGS[@]}" "${CONFIG}"
