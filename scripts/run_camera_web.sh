#!/usr/bin/env bash
# 实机相机识别预览与无损标定抓拍启动脚本（camera_web_test）。
# 用法示例：
#   ./scripts/run_camera_web.sh
#   ./scripts/run_camera_web.sh -c configs/standard3.yaml -p 8090

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

CONFIG="configs/standard3.yaml"
PORT="8090"

usage() {
  cat <<EOF
用法: $(basename "$0") [选项]

选项:
  -c, --config FILE   配置文件（默认: configs/standard3.yaml）
  -p, --port PORT     WebViewer 端口（默认: 8090）
  -h, --help          显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

require_binary "camera_web_test"
setup_vision_env

LOCAL_IP="$(detect_local_ip)"
echo "========================================================"
echo "  相机 Web 预览 (camera_web_test)"
echo "  配置文件  : ${CONFIG}"
echo "  WebViewer : http://${LOCAL_IP}:${PORT}"
echo "  无损抓拍  : curl 'http://${LOCAL_IP}:${PORT}/api/snap?n=1&prefix=shot'"
echo "========================================================"

cd "${ROOT_DIR}"
exec "${BUILD_DIR}/camera_web_test" "--port=${PORT}" "${CONFIG}"
