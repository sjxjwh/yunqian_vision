#!/usr/bin/env bash
# 离线单元测试一键运行脚本（不依赖相机 SDK、串口或仿真器）。
# 用法示例：
#   ./scripts/run_tests.sh                   # 运行全部 4 项核心单测
#   ./scripts/run_tests.sh --build           # 先编译再运行
#   ./scripts/run_tests.sh shoot_on_move_test uvl_tracker_test

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

DO_BUILD=0
SELECTED=()
ALL_TESTS=(
  shoot_on_move_test
  uvl_tracker_test
  aim_corrector_test
  sim_camera_model_test
)

usage() {
  cat <<EOF
用法: $(basename "$0") [选项] [测试名...]

默认测试列表: ${ALL_TESTS[*]}

选项:
  --build       运行前先编译选中测试
  -h, --help    显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) SELECTED+=("$1"); shift ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_TESTS[@]}")
fi

if [[ "${DO_BUILD}" -eq 1 ]]; then
  "${ROOT_DIR}/scripts/build.sh" "${SELECTED[@]}"
fi

setup_vision_env
cd "${ROOT_DIR}"

PASSED=()
FAILED=()

for t in "${SELECTED[@]}"; do
  require_binary "${t}"
  echo "--------------------------------------------------------"
  echo "[test] 运行 ${t} ..."
  echo "--------------------------------------------------------"
  if "${BUILD_DIR}/${t}"; then
    PASSED+=("${t}")
  else
    FAILED+=("${t}")
  fi
done

echo "========================================================"
echo "  测试汇总: 通过 ${#PASSED[@]} / $((${#PASSED[@]} + ${#FAILED[@]}))"
if [[ ${#PASSED[@]} -gt 0 ]]; then
  echo "  [PASS] ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  [FAIL] ${FAILED[*]}"
  echo "========================================================"
  exit 1
fi
echo "========================================================"
