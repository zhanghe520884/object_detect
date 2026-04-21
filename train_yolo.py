import os
from ultralytics import YOLO


def run_compare_experiments():
    # 1. 定义实验配置映射
    experiments = {
        "YOLO_VOC_Balanced": "configs/voc_train.yaml",
        "YOLO_COCO_Balanced": "configs/coco_train.yaml"
    }

    for exp_name, cfg_path in experiments.items():
        print(f"\n{'=' * 20} 正在开始实验: {exp_name} {'=' * 20}")

        # 2. 初始化模型
        model = YOLO("yolov8s.pt")

        # 3. 启动训练
        model.train(
            data=cfg_path,
            epochs=30,
            imgsz=640,
            amp=False,
            batch=16,
            workers = 1,
            name=exp_name,
            device=0,
            save=True,
            project="runs/train_results"
        )
        print(f"{exp_name} 训练完成")


if __name__ == '__main__':
    run_compare_experiments()