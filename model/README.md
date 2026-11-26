# model/

本目录用于存放板端 RKNN 模型文件（`*.rknn`）。

模型权重属于训练产物，**不纳入 Git 版本管理**，因此克隆仓库后这里是空的（只保留本说明）。
按下文流程在 x86 主机上转换得到模型后，把文件放到本目录，`edge_rtsp_service.py` 会扫描
本目录下的全部 `*.rknn` 并通过 `GET /api/models` 暴露给控制台：

```bash
# 板端（默认模型目录就是本目录）
python edge_rtsp_service.py --model-dir ./model --host 0.0.0.0 --port 8090
```

模型命名建议保留精度与构建信息，例如：

```text
landslide_yolov8n_int8_rk3576_split.rknn
landslide_yolov8n_int8_rk3576_split_v2.rknn
```

`*.rknn` 已在 `.gitignore` 中排除，`git add -A` 不会误提交权重。
