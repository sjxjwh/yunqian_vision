#!/usr/bin/env bash
# 快捷编译脚本（自动识别本机 .deps/toolchain.cmake 与板端系统依赖）。
# 用法示例：
#   ./scripts/build.sh sim        # 编译仿真闭环 + 离线单元测试
#   ./scripts/build.sh real       # 编译实机程序（standard_mpc / camera_web_test / real_gimbal_sysid）
#   ./scripts/build.sh tests      # 编译离线单元测试
#   ./scripts/build.sh all        # 编译全部 target
#   ./scripts/build.sh sim_standard uvl_tracker_test  # 编译指定 target

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

JOBS="$(nproc)"
RECONFIGURE=0
TARGETS=()

usage() {
  cat <<EOF
用法: $(basename "$0") [选项] [预设|target...]

预设:
  sim     编译 sim_standard + 4 个离线单元测试（默认）
  real    编译 standard_mpc、camera_web_test、real_gimbal_sysid
  tests   编译 shoot_on_move_test、uvl_tracker_test、aim_corrector_test、sim_camera_model_test
  all     编译全部 CMake target

选项:
  -j, --jobs N        并行编译线程数（默认: $(nproc)）
  --reconfigure       强制重新运行 cmake 配置
  -h, --help          显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2 ;;
    --reconfigure) RECONFIGURE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) TARGETS+=("$1"); shift ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=("sim")
fi

# 展开预设名
BUILD_ALL=0
EXPANDED=()
for t in "${TARGETS[@]}"; do
  case "$t" in
    sim)
      EXPANDED+=(sim_standard shoot_on_move_test uvl_tracker_test aim_corrector_test sim_camera_model_test)
      ;;
    real)
      EXPANDED+=(standard_mpc camera_web_test real_gimbal_sysid)
      ;;
    tests)
      EXPANDED+=(shoot_on_move_test uvl_tracker_test aim_corrector_test sim_camera_model_test)
      ;;
    all)
      BUILD_ALL=1
      ;;
    *)
      EXPANDED+=("$t")
      ;;
  esac
done

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" || "${RECONFIGURE}" -eq 1 ]]; then
  CMAKE_ARGS=(-S "${ROOT_DIR}" -B "${BUILD_DIR}")
  if [[ -f "${ROOT_DIR}/.deps/toolchain.cmake" ]]; then
    echo "[build] 检测到 .deps/toolchain.cmake，使用本地依赖配置 CMake..."
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/.deps/toolchain.cmake")
  else
    echo "[build] 使用系统依赖配置 CMake..."
  fi
  cmake "${CMAKE_ARGS[@]}"
fi

if [[ "${BUILD_ALL}" -eq 1 ]]; then
  echo "[build] 编译全部 target (-j${JOBS})..."
  cmake --build "${BUILD_DIR}" -j"${JOBS}"
else
  echo "[build] 编译 target: ${EXPANDED[*]} (-j${JOBS})..."
  cmake --build "${BUILD_DIR}" --target "${EXPANDED[@]}" -j"${JOBS}"
fi

echo "[build] 完成。"
