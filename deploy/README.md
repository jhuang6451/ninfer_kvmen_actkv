# deploy/ — 容器化构建与运行（Podman，RTX 50 / sm_120a）

| 文件 | 作用 |
|---|---|
| `Dockerfile.builder` | 编译环境镜像（CUDA 13.3 devel），tag 建议 `localhost/ninfer-builder:cuda13.3` |
| `Dockerfile.runner` | 运行镜像（CUDA 13.3 runtime），需要先把编译产物放到构建上下文的 `dist/bin/{ninfer,ninfer-serve}` |
| `convert-model.sh` | 在 builder 容器内把 GGUF（+mmproj）转成 `.ninfer` |
| `start-ninfer-server.sh` | 用 `podman run` 前台启动服务（KVMem + ActKV 参数） |
| `ninfer-actkv.container` | 同样参数的 Quadlet 单元（systemd 用户服务） |

## 使用顺序

1. 在仓库根目录准备带补丁的后端：`python3 scripts/prepare-backends.py`
   （`backends/ninfer` 钉在 `versions.json` 的 base_commit，再应用 `backends/patches/ninfer-kvmem.patch`）。
2. 构建 builder 镜像并编译，把 `ninfer`、`ninfer-serve` 放进 `dist/bin/`，再构建 runner 镜像
   `localhost/ninfer-kvmem-actkv:rtx50`。
3. `./convert-model.sh [模型目录]`（或设置 `MODELS_DIR`）。
4. `./start-ninfer-server.sh [模型.ninfer] [端口]`（或设置 `MODEL_PATH`）。

`--kvmem-action-lrfu` 即 ActKV 开关。模型路径默认 `$HOME/Models/Qwen3.8-27B`，按需覆盖。
