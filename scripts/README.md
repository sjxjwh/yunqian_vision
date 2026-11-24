# 常用启动脚本速查

所有脚本均可从仓库任意目录调用，会自动配置本地 `.deps/`、OpenVINO/TBB、海康 MVS 及仿真 Rust 动态库路径（由 `_env.sh` 统一处理）。

| 脚本 | 用途 | 常用示例 |
|---|---|---|
| `scripts/run_sim.sh` | **仿真闭环一键启动**（`daedalus` + `sim_standard`，`Ctrl+C` 同时退出） | `./scripts/run_sim.sh`<br>`./scripts/run_sim.sh --profile`<br>`./scripts/run_sim.sh --vision-only` |
| `scripts/run_standard.sh` | **实机自瞄启动**（`standard_mpc`，默认 `--no-fire --web` 安全调试） | `./scripts/run_standard.sh`<br>`./scripts/run_standard.sh --live`<br>`./scripts/run_standard.sh --live --web` |
| `scripts/run_camera_web.sh` | **相机 Web 预览与无损抓拍**（`camera_web_test`） | `./scripts/run_camera_web.sh`<br>`./scripts/run_camera_web.sh -p 8091` |
| `scripts/run_tests.sh` | **离线单元测试**（4 项核心单测，不依赖硬件） | `./scripts/run_tests.sh`<br>`./scripts/run_tests.sh --build` |
| `scripts/build.sh` | **快捷编译**（自动识别 `.deps/toolchain.cmake` 或系统依赖） | `./scripts/build.sh sim`<br>`./scripts/build.sh real`<br>`./scripts/build.sh sim_standard` |
| `scripts/deploy_orin.sh` | **同步到 Orin NX**（默认 `yq11@192.168.55.1`，可选远程编译） | `./scripts/deploy_orin.sh`<br>`./scripts/deploy_orin.sh --build`<br>`./scripts/deploy_orin.sh --dry-run` |
| `tools/run_real_sysid.sh` | **实车云台系统辨识**（已有脚本，带限位安全确认与自动拟合） | `./tools/run_real_sysid.sh configs/standard3.yaml` |

任意脚本加 `-h` / `--help` 可查看完整参数说明。
