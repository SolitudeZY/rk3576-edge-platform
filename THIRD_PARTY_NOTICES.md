# 第三方组件说明

本仓库只保存自有源代码；以下组件以**运行时依赖**或**前端静态资源**形式使用，权重与二进制均未入库。

| 组件 | 关系 | 许可证 | 说明 |
|---|---|---|---|
| [hls.js](https://github.com/video-dev/hls.js) | 随仓库分发（`stream_lab/web/hls.min.js`） | Apache-2.0 | MediaMTX 低延迟 HLS 播放所需的浏览器端播放器库，已压缩为单文件 |
| [MediaMTX](https://github.com/bluenviron/mediamtx) | 运行时依赖 | MIT | RTSP/HLS 中转服务；仓库不包含 `mediamtx.exe`，需自行下载或使用 Docker 镜像 |
| [FFmpeg](https://ffmpeg.org/) | 运行时依赖 | LGPL-2.1+/GPL（取决于构建） | 用于循环推流与板端 H.264 编码；仓库不包含二进制 |
| [RKNN-Toolkit2 / RKNN-Toolkit-Lite2](https://github.com/airockchip/rknn-toolkit2) | 运行时依赖 | Rockchip SDK 许可 | 模型转换与板端推理运行时，需按 Rockchip 官方协议安装 |
| YOLOv8 模型图 | 训练产物 | 由训练方决定 | 仓库不含 `.pt` / `.onnx` / `.rknn` 权重 |

`stream_lab/bin/` 目录（MediaMTX 可执行文件）在 `.gitignore` 中排除。
