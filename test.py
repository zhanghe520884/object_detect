import torch
import cv2
import os
import numpy as np
from pathlib import Path
from torchvision.transforms import functional as F
from tqdm import tqdm


class UniversalDetector:
    def __init__(self, model_path, model_type='fasterrcnn', device=None):
        self.raw_model_path = model_path
        self.model_path = os.path.abspath(model_path.replace('\\', '/'))

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到权重文件，请检查绝对路径: {self.model_path}")

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" 正在使用设备: {self.device}")

        self.model_type = model_type.upper()
        self.classes = ['bg', 'person', 'bicycle', 'car', 'bus', 'motorcycle']

        if 'YOLO' in self.model_type:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"YOLO 模型已加载")
        else:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn

            self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(self.classes))

            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            print(f"{self.model_type} 模型已加载")

    def predict_frame(self, frame, conf_threshold=0.5):
        """对单帧图像进行检测"""
        if 'YOLO' in self.model_type:
            # YOLO 推理
            results = self.model(frame, conf=conf_threshold, verbose=False)[0]
            return results.plot()
        else:
            # RCNN 系列推理
            img_tensor = F.to_tensor(frame).to(self.device)
            with torch.no_grad():
                prediction = self.model([img_tensor])[0]

            # 绘制结果
            res_img = frame.copy()
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > conf_threshold:
                    b = box.cpu().numpy().astype(int)
                    # 画框
                    cv2.rectangle(res_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    # 写标签
                    label_text = f"{self.classes[label]}: {score:.2f}"
                    cv2.putText(res_img, label_text, (b[0], b[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return res_img

    def _get_save_name(self, source_path):
        path_parts = Path(self.raw_model_path).parts

        dataset_info = "Unknown"
        for part in reversed(path_parts):
            p_up = part.upper()
            if "VOC" in p_up or "COCO" in p_up:
                dataset_info = part
                break

        source_stem = Path(source_path).stem
        source_ext = Path(source_path).suffix

        return f"{source_stem}_{self.model_type}_{dataset_info}{source_ext}"

    def process_source(self, source_path, save_dir='test_results'):
        if not os.path.exists(source_path):
            print(f"找不到输入文件: {source_path}")
            return

        os.makedirs(save_dir, exist_ok=True)
        new_filename = self._get_save_name(source_path)
        out_path = os.path.join(save_dir, new_filename)

        is_image = source_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

        if is_image:
            frame = cv2.imread(source_path)
            if frame is None:
                print(f"图片读取失败: {source_path}")
                return
            result = self.predict_frame(frame)
            cv2.imwrite(out_path, result)
            print(f"图片处理完成: {out_path}")

        else:
            # 视频处理
            cap = cv2.VideoCapture(source_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            with tqdm(total=total_frames, desc=f"正在处理 {new_filename}", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    res_frame = self.predict_frame(frame)
                    out.write(res_frame)
                    pbar.update(1)

            cap.release()
            out.release()
            print(f"视频处理完成: {out_path}")


# --- 运行入口 ---
if __name__ == "__main__":

    # 示例 A: 测试 YOLO 模型
    # WEIGHT = 'runs/detect/runs/train_results/YOLO_COCO_Balanced-2/weights/best.pt'
    # TYPE = 'YOLO'

    # 示例 B: 测试 FasterRCNN 模型
    WEIGHT = 'runs/rcnn_results/FasterRCNN_COCO/best.pth'
    TYPE = 'FasterRCNN'

    # 初始化探测器
    detector = UniversalDetector(model_path=WEIGHT, model_type=TYPE)

    # 输入目标 (图片或视频)
    TARGET = "test/img_3.png"

    detector.process_source(TARGET)