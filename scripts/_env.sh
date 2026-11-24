#!/usr/bin/env bash
# 公共环境初始化脚本（由其他启动脚本 source，不要单独执行）。
# 兼容本机 x86_64（含 .deps 本地依赖）以及 NUC / Orin NX 板端环境。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

_prepend_ld_path() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${dir}:"*) ;;
      *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
}

setup_vision_env() {
  # 本机工作区 .deps（无系统级依赖时使用）
  _prepend_ld_path "${ROOT_DIR}/.deps/install/lib"
  local ov_dir=""
  for ov_dir in "${ROOT_DIR}"/.deps/l_openvino_toolkit_*; do
    if [[ -d "${ov_dir}" ]]; then
      _prepend_ld_path "${ov_dir}/runtime/lib/intel64"
      _prepend_ld_path "${ov_dir}/runtime/3rdparty/tbb/lib"
      break
    fi
  done

  # 系统级 OpenVINO（NUC）
  for ov_dir in /opt/intel/openvino_*; do
    if [[ -d "${ov_dir}" ]]; then
      _prepend_ld_path "${ov_dir}/runtime/lib/intel64"
      _prepend_ld_path "${ov_dir}/runtime/3rdparty/tbb/lib"
      break
    fi
  done

  # 海康 MVS SDK（Orin NX aarch64 / x86_64）
  if [[ "$(uname -m)" == "aarch64" ]]; then
    _prepend_ld_path "/opt/MVS/lib/aarch64"
  else
    _prepend_ld_path "/opt/MVS/lib/64"
  fi
}

setup_sim_rust_env() {
  _prepend_ld_path "${ROOT_DIR}/.rust/target/release"
  local std_so=""
  std_so="$(find "${ROOT_DIR}/.rust/rustup" -maxdepth 6 -name 'libstd-*.so' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${std_so}" ]]; then
    _prepend_ld_path "$(dirname "${std_so}")"
  fi
}

require_binary() {
  local name="$1"
  local bin="${BUILD_DIR}/${name}"
  if [[ ! -x "${bin}" ]]; then
    echo "[错误] 未找到可执行文件: ${bin}" >&2
    echo "请先编译: ./scripts/build.sh ${name}" >&2
    exit 1
  fi
}

detect_local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
}
