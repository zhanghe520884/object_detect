import os
import torch
import json
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import confusion_matrix


# --- 1. 通用数据集类---
class UniversalRCNNDataset(Dataset):
    def __init__(self, root, is_coco=False, transforms=None):
        self.root = root
        self.is_coco = is_coco
        self.transforms = transforms
        self.img_dir = os.path.join(root, "images")
        self.class_map = {
            'person': 1, 'bicycle': 2, 'car': 3, 'bus': 4,
            'motorcycle': 5, 'motorbike': 5
        }

        if is_coco:
            ann_dir = os.path.join(root, "annotations")
            json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
            if not json_files: raise FileNotFoundError(f"No JSON in {ann_dir}")
            with open(os.path.join(ann_dir, json_files[0]), 'r') as f:
                data = json.load(f)

            self.imgs = data['images']
            self.coco_id_to_name = {int(cat['id']): cat['name'] for cat in data['categories']}

            self.ann_map = {int(img['id']): [] for img in data['images']}
            for ann in data['annotations']:
                img_id = int(ann['image_id'])
                if img_id in self.ann_map:
                    self.ann_map[img_id].append(ann)
        else:
            self.ann_dir = os.path.join(root, "annotations")
            self.imgs = [f for f in sorted(os.listdir(self.img_dir)) if f.endswith(('.jpg', '.png'))]

    def __getitem__(self, idx):
        boxes, labels = [], []
        if self.is_coco:
            img_info = self.imgs[idx]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            img = Image.open(img_path).convert("RGB")

            for ann in self.ann_map.get(int(img_info['id']), []):
                raw_id = int(ann['category_id'])
                name = self.coco_id_to_name.get(raw_id, "").lower()

                if name in self.class_map:
                    x, y, w, h = ann['bbox']
                    if w > 1 and h > 1:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(self.class_map[name])
        else:
            img_name = self.imgs[idx]
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0] + ".xml")

            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                for obj in tree.findall("object"):
                    name = obj.find("name").text.lower().strip()
                    if name in self.class_map:
                        labels.append(self.class_map[name])
                        b = obj.find("bndbox")
                        boxes.append([float(b.find('xmin').text), float(b.find('ymin').text),
                                      float(b.find('xmax').text), float(b.find('ymax').text)])

        if len(boxes) == 0:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros(0, dtype=torch.int64), "image_id": torch.tensor([idx])}
        else:
            target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                      "labels": torch.as_tensor(labels, dtype=torch.int64), "image_id": torch.tensor([idx])}

        if self.transforms: img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


# --- 2. 预测结果可视化 ---
def save_test_predictions(model, device, loader, classes, out_dir, num_samples=3):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    target_h = 480

    with torch.no_grad():
        data_iter = iter(loader)
        for i in range(num_samples):
            try:
                images, targets = next(data_iter)
                img_tensor = images[0].to(device)
                outputs = model([img_tensor])[0]

                img_cv = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                # 预测框 (红色)
                for box, label, score in zip(outputs['boxes'].cpu().numpy(),
                                             outputs['labels'].cpu().numpy(),
                                             outputs['scores'].cpu().numpy()):
                    if score > 0.4:  # 置信度阈值
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(img_cv, p1, p2, (0, 0, 255), 2)
                        txt = f"{classes[label]} {score:.2f}"
                        cv2.putText(img_cv, txt, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 真值框 (绿色)
                for box, label in zip(targets[0]['boxes'].cpu().numpy(), targets[0]['labels'].cpu().numpy()):
                    if label == 0: continue
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(img_cv, p1, p2, (0, 255, 0), 1)

                cv2.imwrite(f"{out_dir}/test_predict_{i}.jpg", img_cv)
            except StopIteration:
                break


# --- 3. 训练 ---
def run_model_experiment(train_path, is_coco, model_type, dataset_name):
    exp_name = f"{model_type}_{dataset_name}"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    out_dir = f"runs/rcnn_results/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    classes = ['bg', 'person', 'bicycle', 'car', 'bus', 'motorcycle']

    # 模型加载
    if model_type == "FasterRCNN":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif model_type == "FastRCNN":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, rpn_pre_nms_top_n_train=2000)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=6)

    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 6)
    model.to(device)

    ds = UniversalRCNNDataset(train_path, is_coco=is_coco, transforms=torchvision.transforms.ToTensor())
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=0.0005)

    best_loss = float('inf')

    for epoch in range(5):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"{exp_name} Ep {epoch}")
        for imgs, tars in pbar:
            valid_imgs, valid_tars = [], []
            for im, ta in zip(imgs, tars):
                if ta['labels'].shape[0] > 0:
                    valid_imgs.append(im.to(device));
                    valid_tars.append({k: v.to(device) for k, v in ta.items()})

            if not valid_imgs: continue

            loss_dict = model(valid_imgs, valid_tars)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad();
            losses.backward();
            optimizer.step()
            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        # 权重保存
        avg_loss = epoch_loss / len(loader)
        torch.save(model.state_dict(), f"{out_dir}/last.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{out_dir}/best.pth")

    return model, device


# --- 4. 混淆矩阵---
def generate_norm_cm(model, device, exp_name):
    test_ds = UniversalRCNNDataset("dataset/voc_split_mini/test", is_coco=False,
                                   transforms=torchvision.transforms.ToTensor())
    loader = DataLoader(test_ds, batch_size=1, collate_fn=lambda x: tuple(zip(*x)))

    # 生成测试集预测预览
    classes = ['bg', 'person', 'bicycle', 'car', 'bus', 'motorcycle']
    save_test_predictions(model, device, loader, classes, f"runs/rcnn_results/{exp_name}/test_previews")

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for imgs, tars in tqdm(loader, desc=f"Eval {exp_name}"):
            if tars[0]['labels'].shape[0] == 0: continue

            true_label = tars[0]['labels'][0].item()
            outs = model([imgs[0].to(device)])[0]

            if len(outs['labels']) > 0 and outs['scores'][0] > 0.3:
                pred_label = outs['labels'][0].item()
            else:
                pred_label = 0

            all_p.append(pred_label);
            all_t.append(true_label)

    cm = confusion_matrix(all_t, all_p, labels=range(6))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(10, 8));
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=classes, yticklabels=classes)
    plt.savefig(f"runs/rcnn_results/{exp_name}/confusion_matrix_final.png");
    plt.close()


if __name__ == "__main__":
    model_configs = ["FasterRCNN", "FastRCNN", "RCNN"]
    dataset_configs = [("dataset/voc_split_mini/train", False, "VOC"), ("dataset/coco_split_mini/train", True, "COCO")]

    for m_type in model_configs:
        for d_path, is_coco, d_name in dataset_configs:
            m, dev = run_model_experiment(d_path, is_coco, m_type, d_name)
            generate_norm_cm(m, dev, f"{m_type}_{d_name}")