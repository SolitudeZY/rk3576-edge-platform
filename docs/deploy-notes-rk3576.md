# RK3576 deployment and benchmark

This directory is the RK3576 replacement for `ZY/rk3568_deploy`. It keeps the
old monolithic YOLOv8 output compatible and also accepts the 6/9-output head
used by Rockchip's optimized YOLOv8 model.

The supported software set is:

- RKNN-Toolkit2: 2.3.2
- RKNN-Toolkit-Lite2: 2.3.2
- `librknnrt.so`: 2.3.2
- model target: `rk3576`
- board architecture: AArch64

## 1. Convert on the host

Conversion runs in the x86 RKNN-Toolkit2 environment, not in RKNNLite on the
board. INT8 is the default deployment precision because FP16 performance is
substantially lower for YOLO at 640x640.

Create a calibration list containing representative positive and negative
images. Use absolute paths so Toolkit2 can resolve every entry:

```bash
find /absolute/path/to/calibration_images -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  | head -n 200 > calibration.txt
```

The current training export has one output shaped `[1,5,8400]`. Do not feed it
directly to an INT8 build: coordinates and scores would share one output scale.
Split it first:

```bash
python ZY/rk3576_deploy/split_yolo_output.py \
  --source ZY/script_yolo/runs/train/landslide_yolov8n_640/weights/best.onnx \
  --output ZY/rk3576_deploy/landslide_yolov8n_split.onnx
```

The split model has independent outputs `[1,4,8400]` and `[1,1,8400]`.
Convert that model to INT8:

```bash
python ZY/rk3576_deploy/convert_onnx_to_rknn.py \
  --onnx ZY/rk3576_deploy/landslide_yolov8n_split.onnx \
  --output ZY/rk3576_deploy/landslide_yolov8n_int8_rk3576_split.rknn \
  --precision int8 \
  --dataset calibration.txt
```

For an accuracy-only comparison, build FP16 explicitly:

```bash
python ZY/rk3576_deploy/convert_onnx_to_rknn.py \
  --onnx best.onnx \
  --output landslide_yolov8n_fp16_rk3576.rknn \
  --precision fp16
```

If the converter prints the monolithic-output warning, the model remains
usable, but it is not the fastest RKNN YOLO graph. For peak throughput, export
with Rockchip's `airockchip/ultralytics_yolov8` fork so the three detection
heads remain separate. `infer_rknn.py` supports both layouts.

## 2. Copy to the RK3576 board

From Windows PowerShell, replace the user and IP address as needed:

```powershell
scp -r E:\Python_Prj\landSlide\ZY\rk3576_deploy `
  kickpi@192.168.1.100:/home/kickpi/
```

The model file is included only if it has already been generated in this
directory.

On the RK3576 board, deployment models are stored under:

```text
/home/kickpi/landslide/rk3576_deploy/model/
```

The edge API exposes `GET /api/models`. The stream-lab Web console forwards the
selected filename when starting the pipeline, so switching between models does
not require restarting the systemd service.

## 3. Prepare the board environment

Check that the runtime stack is aligned before measuring performance:

```bash
python -m pip show rknn-toolkit-lite2
sha256sum /usr/lib/librknnrt.so
cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
```

The official v2.3.2 `librknnrt.so` SHA256 is:

```text
d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8
```

Use the existing RKNN virtual environment. If OpenCV and NumPy are missing:

```bash
source ~/venvs/landslide-rknn/bin/activate
python -m pip install -r ~/rk3576_deploy/requirements-board.txt
```

On Ubuntu 24, a virtual environment created with `--system-site-packages` can
also reuse the board vendor's `python3-opencv` package.

## 4. Benchmark pure RKNN inference

Benchmark after ten warmup runs and compare all RK3576 core masks:

```bash
cd ~/rk3576_deploy
python benchmark_rknn.py \
  --model landslide_yolov8n_int8_rk3576_split.rknn \
  --image test.jpg \
  --warmup 10 \
  --runs 100 \
  --cores auto,core0,core1,both
```

The reported time covers only `runtime.inference()`. It excludes image resize,
color conversion, YOLO decode, NMS, drawing, and file writing. Use the fastest
stable core mode for production; `both` is not guaranteed to beat one core for
every model.

Rockchip's published YOLOv8n INT8 RK3576 single-core reference is about 90.2
FPS at maximum NPU frequency. A custom graph can differ, but a result near 8
FPS indicates FP16, an unoptimized graph, low NPU frequency, or inefficient
operators and should not be treated as normal INT8 performance.

Inspect the current NPU frequency if results are unexpectedly low:

```bash
for d in /sys/class/devfreq/*npu*; do
  echo "=== $d ==="
  cat "$d/governor" "$d/cur_freq" "$d/min_freq" "$d/max_freq" 2>/dev/null
done
```

For a temporary controlled benchmark, switch the NPU governor to performance:

```bash
echo performance | sudo tee /sys/class/devfreq/*npu*/governor
```

## 5. Image and video inference

Image:

```bash
python infer_rknn.py \
  --model landslide_yolov8n_int8_rk3576_split.rknn \
  --source test.jpg \
  --output result.jpg \
  --cores auto \
  --warmup 5
```

Video, camera, and RTSP:

```bash
python infer_rknn.py --model model.rknn --source test.mp4 --output result.mp4
python infer_rknn.py --model model.rknn --source 0 --output camera.mp4
python infer_rknn.py --model model.rknn --source 'rtsp://...' --output rtsp.mp4
```

The inference script prints separate preprocessing, RKNN inference, and
postprocessing timings. Video drawing and encoding are deliberately excluded
from `pipeline_fps`.

The static-shape dynamic-range warning is harmless for this 640x640 model:

```text
query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type
```

## 6. Local tests

The unit tests do not require an RK3576 board or RKNNLite:

```bash
cd ZY/rk3576_deploy
python -m unittest -v test_rk3576_deploy.py
```
