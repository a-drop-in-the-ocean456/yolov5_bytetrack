import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# 添加 YOLOv5 和 ByteTrack 路径
sys.path.insert(0, './yolov5')
sys.path.insert(0, './ByteTrack')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

# ByteTrack tracker
from yolox.tracker.byte_tracker import BYTETracker

class BYTETrackerArgs:
    """ByteTrack 参数配置"""
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

def preprocess(img, img_size=640):
    """预处理图像"""
    img0 = img.copy()
    img = letterbox(img, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0

def run_inference(video_path, output_path):
    """运行 YOLOv5s + ByteTrack 推理"""
    
    # 初始化设备
    # device = select_device('cpu')
    device = select_device('0')
    
    # 加载 YOLOv5s 模型
    print("加载 YOLOv5s 模型...")
    model = DetectMultiBackend('yolov5s.pt', device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    
    # 初始化 ByteTrack
    print("初始化 ByteTrack...")
    tracker = BYTETracker(BYTETrackerArgs(), frame_rate=30)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_id = 0
    
    # 定义要跟踪的类别 (car, truck, bus)
    target_classes = [2, 5, 7]  # COCO dataset: car, bus, truck
    
    print("开始推理...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # 预处理
        img, img0 = preprocess(frame)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        # YOLOv5 推理
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=target_classes, max_det=1000)
        
        # 处理检测结果
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    detections.append([x1, y1, x2, y2, float(conf)])
        
        # ByteTrack 跟踪
        if len(detections) > 0:
            detections_np = np.array(detections)
            online_targets = tracker.update(detections_np, [height, width], [height, width])
            
            # 绘制跟踪结果
            for track in online_targets:
                tlwh = track.tlwh
                track_id = track.track_id
                x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
                x2, y2 = x1 + w, y1 + h
                
                # 绘制边界框
                color = (0, 255, 0)
                cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                
                # 绘制 ID
                label = f'ID:{track_id}'
                cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 写入输出视频
        out.write(img0)
        
        # 显示进度
        if frame_id % 30 == 0:
            print(f"处理进度: {frame_id}/{total_frames} ({100*frame_id/total_frames:.1f}%)")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"推理完成! 输出保存至: {output_path}")

if __name__ == '__main__':
    video_path = 'traffic_video.mp4'
    output_path = 'output/output_tracked.mp4'
    
    run_inference(video_path, output_path)
