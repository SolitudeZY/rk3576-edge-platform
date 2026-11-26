# RK3576 RTSP 实时检测实验台

链路：本机上传视频并循环推送 RTSP，RK3576 拉流完成 INT8 检测后将画面推回本机，网页通过低延迟 HLS 播放检测结果并轮询板端指标。

## 地址

- 输入流：`rtsp://10.8.30.29:8554/input`
- 检测流：`rtsp://10.8.30.29:8554/detected`
- 检测预览：`http://10.8.30.29:8888/detected/index.m3u8`
- 板端 API：`http://10.8.30.54:8090`
- 本机控制台：`http://10.8.30.29:8080`

板端所有 RKNN 文件统一放在 `/home/kickpi/landslide/rk3576_deploy/model/`。
Web 控制台的“RKNN 模型”下拉框读取板端模型列表，启动视频或上传单图时使用当前选中的模型。

浏览器不能直接播放 RTSP，因此本机 MediaMTX 同时负责 RTSP 中转和 HLS 转换。

## 本机启动

要求：

- `stream_lab/bin/mediamtx.exe`
- `D:\ffmpeg\bin\ffmpeg.exe`
- Python 3.10 或更新版本

PowerShell 执行：

```powershell
cd E:\Python_Prj\landSlide\ZY\rk3576_deploy\stream_lab
.\start_stream_lab.ps1
```

然后打开 `http://10.8.30.29:8080`，上传或选择视频并启动链路。

要让服务启动后自动循环推流，并在源 FFmpeg 退出后自动重启：

```powershell
.\start_stream_lab.ps1 -AutoStart -AutoStartVideo test.mp4
```

不指定 `-AutoStartVideo` 时，会选择 `uploads` 目录中最近修改的视频。视频源使用 `-stream_loop -1`，检测输出持续发布到固定 RTSP 路径。

## 局域网访问

本机服务监听 `0.0.0.0`，局域网用户访问：

- Web 控制台：`http://10.8.30.29:8080`
- 检测 HLS：`http://10.8.30.29:8888/detected/index.m3u8`
- 检测 RTSP：`rtsp://10.8.30.29:8554/detected`

如果其他电脑无法连接，需要在 Windows 防火墙放行 TCP `8080`、`8554`、`8888`（以及可选的 `9997` API 端口）。板端 API 监听 `0.0.0.0:8090`，一般只让本机代理访问即可。

## 单图检测

网页的“单图检测”上传入口会把图片转发给 RK3576 的 INT8 模型，返回带检测框的 JPEG 结果。接口限制为 JPG/PNG/WEBP，最大 16 MB；单图请求使用独立的临时 RKNN runtime，不会停止持续视频管线。

## 板端服务

板端文件位于 `/home/kickpi/landslide/rk3576_deploy`。检查服务：

```bash
sudo systemctl status landslide-edge-stream
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/api/metrics
```

查看实时日志：

```bash
journalctl -u landslide-edge-stream -f
```

查看模型列表：

```bash
curl http://10.8.30.54:8090/api/models
```

切换模型不需要重启 systemd。选择模型后再次点击“启动实时链路”，旧 runtime 会释放，新模型会加载；切换期间检测流会短暂重建。

## 指标说明

- CPU、系统内存来自 `psutil`。
- NPU 核心负载与频率来自 `/sys/kernel/debug/rknpu`。
- 当前驱动没有提供独立 NPU 内存统计；页面中的“RKNN 进程内存”是推理服务 RSS 估计值，不等同于专用 NPU 内存。
- `capture-to-publish` 是板端从取得最新帧到写入编码器的延迟，不包含 HLS 分片与浏览器缓冲时间。
