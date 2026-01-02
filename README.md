# YOLOv5s + ByteTrack 汽车视频跟踪项目

本项目展示了如何结合 **YOLOv5s** 目标检测模型和 **ByteTrack** 多目标跟踪算法，对汽车交通视频进行实时推理和跟踪。

## 项目结构

- `yolov5_bytetrack_inference.py`: 核心推理脚本，集成了检测与跟踪逻辑。
- `traffic_video.mp4`: 原始输入视频素材。
- `output_tracked.mp4`: 推理后的结果视频（包含边界框和跟踪 ID）。
- `preview_frames/`: 包含 5 张推理结果的关键帧预览图。
- `README.md`: 本说明文档。

## 环境配置

要运行此项目，您需要安装以下依赖：

### 1. 克隆必要仓库
```bash
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/ifzhang/ByteTrack.git
```

### 2. 安装依赖
```bash
# 安装 YOLOv5 依赖
pip install -r yolov5/requirements.txt

# 安装 ByteTrack 依赖
pip install onnxruntime cython_bbox lap loguru opencv-python
```

### 3. 修复 ByteTrack 兼容性问题 (针对 NumPy 1.20+)
由于 ByteTrack 代码较旧，在较新版本的 NumPy 中会报错。请运行以下命令修复：
```bash
cd ByteTrack
find . -name "*.py" -type f -exec sed -i 's/np\.float/np.float64/g' {} \;
sed -i 's/np\.float6432/np.float32/g' yolox/utils/visualize.py
```

## 如何运行

1. 确保 `yolov5` 和 `ByteTrack` 文件夹与 `yolov5_bytetrack_inference.py` 在同一目录下。
2. 运行推理脚本：
   ```bash
   python yolov5_bytetrack_inference.py
   ```
3. 推理完成后，结果将保存为 `output_tracked.mp4`。

## 推理脚本说明

脚本会自动执行以下步骤：
1. 加载预训练的 `yolov5s.pt` 模型。
2. 初始化 `ByteTrack` 跟踪器。
3. 逐帧读取视频，检测车辆（汽车、公交车、卡车）。
4. 使用 ByteTrack 关联检测框并分配唯一 ID。
5. 将标注后的帧保存到输出视频中。

## 注意事项

- 本项目默认在 **CPU** 上运行。如果您有 NVIDIA GPU，可以修改脚本中的 `device` 参数为 `'0'` 以获得更快的推理速度。
- 如果您更换了视频文件，请在脚本末尾修改 `video_path` 和 `output_path`。
"# yolov5_bytetrack" 
"# yolov5_bytetrack" 
# yolov5_bytetrack
