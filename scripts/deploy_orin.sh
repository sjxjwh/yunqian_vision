#!/usr/bin/env bash
# 同步本仓库代码到三号步兵 Orin NX，并可选远程编译。
# 用法示例：
#   ./scripts/deploy_orin.sh                  # 仅同步源码与配置
#   ./scripts/deploy_orin.sh --build          # 同步并在 Orin NX 编译实机程序
#   ./scripts/deploy_orin.sh --dry-run        # 仅预览将同步的文件，不写入远端

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

HOST="${ORIN_HOST:-yq11@192.168.55.1}"
REMOTE_DIR="${ORIN_DIR:-~/yunqian_vision}"
DO_BUILD=0
DRY_RUN=0
BUILD_TARGET="real"

usage() {
  cat <<EOF
用法: $(basename "$0") [选项]

选项:
  --host USER@IP      目标 Orin NX 地址（默认: ${HOST}，也可设环境变量 ORIN_HOST）
  --dir PATH          远端仓库路径（默认: ${REMOTE_DIR}，也可设环境变量 ORIN_DIR）
  --build             同步后在远端执行 ./scripts/build.sh <target>
  --target NAME       配合 --build 使用的目标/预设（默认: real）
  --dry-run           仅预览同步差异，不实际传输
  -h, --help          显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --dir) REMOTE_DIR="$2"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --target) BUILD_TARGET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

RSYNC_OPTS=(
  -avz
  --progress
  --exclude='.git/'
  --exclude='build/'
  --exclude='.deps/'
  --exclude='.rust/'
  --exclude='.sim/'
  --exclude='logs/'
  --exclude='records/'
  --exclude='calib_captures/'
  --exclude='*.engine'
  --exclude='*.log'
  --exclude='__pycache__/'
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

echo "[deploy] 同步 ${ROOT_DIR}/ -> ${HOST}:${REMOTE_DIR}/"
rsync "${RSYNC_OPTS[@]}" "${ROOT_DIR}/" "${HOST}:${REMOTE_DIR}/"

if [[ "${DO_BUILD}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  echo "[deploy] 在 ${HOST} 远程编译 (${BUILD_TARGET})..."
  ssh "${HOST}" "cd ${REMOTE_DIR} && ./scripts/build.sh ${BUILD_TARGET}"
fi

echo "[deploy] 完成。"
