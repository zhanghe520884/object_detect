# 实验二：目标检测模型综合对比实验
---

## 一、实验概述

本实验学习并对比典型目标检测模型（R-CNN、Fast R-CNN、Faster R-CNN、YOLOv8）在 COCO 与 VOC 两个数据集上的训练与检测性能，深入理解各模型的结构差异、算法原理与工程实现特点。

**实验模型**：R-CNN、Fast R-CNN、Faster R-CNN、YOLOv8（n/s）
 **实验数据集**：COCO 2014、Pascal VOC 2012（交通子集）
 **检测类别**：person、bicycle、car、bus、motorcycle（共 5 类）
 **代码文件**：`train_RCNN.py`、`train_yolo.py`、`test.py`、`temp.py`

------

## 二、数据集介绍与预处理

### 2.1 数据集来源

| 数据集    | 类别总数 | 标注格式 | 原始规模               |
| --------- | -------- | -------- | ---------------------- |
| COCO 2014 | 80       | JSON     | 83K 张图片，241MB 标注 |
| VOC 2012  | 20       | XML      | 约 17K 张图片          |

本实验仅保留与交通相关的 5 个类别：**person（1）、bicycle（2）、car（3）、bus（4）、motorcycle（5）**。

### 2.2 数据集格式差异

**VOC 格式（XML）**：每张图片对应一个 XML 文件，记录图片尺寸及每个目标的类别名与边界框坐标（xmin, ymin, xmax, ymax）。

```xml
<annotation>
  <filename>2007_000032.jpg</filename>
  <size><width>500</width><height>281</height></size>
  <object>
    <name>car</name>
    <bndbox>
      <xmin>51</xmin><ymin>67</ymin>
      <xmax>210</xmax><ymax>226</ymax>
    </bndbox>
  </object>
</annotation>
```

**COCO 格式（JSON）**：所有图片的标注集中存储在一个 JSON 文件中，包含 `images`、`annotations`、`categories` 三个主要字段，边界框格式为 `[x, y, width, height]`。

```json
{
  "images": [{"id": 1, "file_name": "000001.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                   "bbox": [100, 150, 200, 120]}],
  "categories": [{"id": 3, "name": "car"}]
}
```

### 2.3 数据统计脚本：`temp.py`

`temp.py` 是一个轻量统计脚本，用于检查各数据集子集的实际图片数量，核心逻辑如下：

```python
DATA_PATHS = {
    "VOC_Train": {"path": "dataset/voc_split_mini/train/annotations", "type": "xml"},
    "COCO_Train": {"path": "dataset/coco_split_mini/train/annotations", "type": "json"},
    ...
}

def count_images():
    for name, config in DATA_PATHS.items():
        if config['type'] == 'xml':
            # VOC：每个 XML 文件对应一张图片，直接统计 XML 文件数
            count = len([f for f in os.listdir(folder_path) if f.endswith('.xml')])

        elif config['type'] == 'json':
            # COCO：解析 JSON 中的 images 列表长度
            with open(json_file, 'r') as f:
                data = json.load(f)
                count = len(data.get('images', []))
```

**设计要点**：VOC 和 COCO 的标注格式完全不同，统计方式也需区分——VOC 是「一文件一图」，COCO 是「一文件含全部图」，此脚本正确处理了这一差异。

### 2.4 数据集处理流程

```
原始数据集
  ├─ COCO: 48,675 张（筛选 5 类后）→ dataset/coco_mini
  └─ VOC:  10,870 张（筛选 5 类后）→ dataset/voc_mini
           ↓ 按 8:2 划分，COCO 缩减至 VOC 的 1.2 倍
  ├─ coco_split / voc_split
  │    训练集: 8,696 张  验证集: 2,174 张  测试集: 2,174 张
           ↓ 再缩减训练集与验证集至 1/3（原单 epoch 需 20+ min）
  └─ coco_split_mini / voc_split_mini（最终训练使用）
```

> VOC 测试集由 COCO 测试集（JSON）转换为 XML 格式，保证两套数据集使用同一测试集，具备可比性。

------

## 三、算法原理详解

### 3.1 R-CNN（Region-based CNN，2014）

#### 核心思想

R-CNN 是第一个将深度学习引入目标检测的里程碑工作，将检测问题拆分为「找框」和「识别框」两个独立步骤。

#### 算法流程

```
输入图像
   ↓
Selective Search（选择性搜索）
   → 提取约 2000 个候选框（Region Proposals）
   ↓
对每个候选框单独：
   ① 强制 Warp 至 227×227
   ② 送入 CNN（AlexNet）提取 4096 维特征
   ↓
SVM 分类器（每类一个 SVM）→ 类别预测
边界框回归器 → 坐标精修
```

#### 关键概念：Selective Search

Selective Search 并非随机采样，而是基于图像分割的层次化合并算法：

1. 用 Felzenszwalb 算法做初始超像素分割
2. 按颜色、纹理、大小、填充度等相似度将相邻区域反复合并
3. 每次合并产生一个新候选框，最终生成约 2000 个多尺度候选框

#### 核心公式

IoU（交并比）是衡量候选框质量的基础指标：
$$
IoU = \frac{|A \cap B|}{|A \cup B|}
$$
正样本定义：IoU ≥ 0.5 的候选框；负样本：IoU < 0.3。

边界框回归目标（学习从候选框 P 到真值框 G 的变换）：
$$
t_x = (G_x - P_x) / P_w, \quad t_y = (G_y - P_y) / P_ht_w = \log(G_w / P_w), \quad t_h = \log(G_h / P_h)
$$

#### 主要缺陷

- **速度极慢**：2000 个候选框各自做一次完整 CNN 前向传播，处理一张图需 47 秒（GPU）
- **存储开销大**：需提前将所有候选框特征缓存到磁盘再训练 SVM
- **流程分离**：CNN 特征提取、SVM 分类、边界框回归三个阶段独立训练，无法端到端优化

#### 本实验模拟实现

由于原版 R-CNN 依赖复杂的 Caffe 环境，本实验用 Faster R-CNN 去掉预训练权重来近似模拟：

```python
# train_RCNN.py - RCNN 模拟
elif model_type == "RCNN":
    # 不加载预训练权重，从随机初始化开始训练
    # 模拟 R-CNN 缺乏预训练特征提取能力的特点
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=6)
```

> **说明**：此模拟仅在「无预训练」这一点上近似 R-CNN，底层仍是 RPN 而非 Selective Search，因此性能数据不能完全代表真实 R-CNN。

------

### 3.2 Fast R-CNN（2015）

#### 核心改进：共享特征图

R-CNN 的最大瓶颈是 2000 次重复 CNN 计算。Fast R-CNN 的解决方案是**先对整图提取一次特征图，再在特征图上处理所有候选框**。

#### 算法流程

```
输入图像
   ↓
CNN 骨干网络（VGG16）→ 特征图（Feature Map）
                             ↑
Selective Search → 候选框坐标 → 映射到特征图上的 ROI 区域
                                    ↓
                               ROI Pooling（统一尺寸）
                                    ↓
                          全连接层 → Softmax 分类
                                  → 边界框回归
```

#### 关键技术：ROI Pooling

ROI Pooling 将任意尺寸的候选区域特征统一池化为固定大小（如 7×7），原理：

1. 将候选框坐标从原图空间映射到特征图空间（除以步长 stride）
2. 将映射后的区域划分为 H×W 个子区域（如 7×7）
3. 对每个子区域做 Max Pooling，输出固定维度特征

```
候选框 [x1,y1,x2,y2] → 特征图坐标 [x1/s, y1/s, x2/s, y2/s]
→ 划分 7×7 格子 → 每格 MaxPool → 7×7 固定特征
```

#### 多任务损失函数

Fast R-CNN 首次实现分类与定位的联合训练：
$$
L = L_{cls}(p, u) + \lambda \cdot [u \geq 1] \cdot L_{loc}(t^u, v)
$$
其中 $L_{cls}$ 为交叉熵分类损失，$L_{loc}$ 为 Smooth L1 定位损失：
$$
L_{loc} = \text{smooth}_{L1}(t^u - v) = \begin{cases} 0.5x^2 & |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}
$$
Smooth L1 相比 L2 对离群点（错误标注的框）更鲁棒。

#### 本实验模拟实现

```python
# train_RCNN.py - Fast R-CNN 模拟
elif model_type == "FastRCNN":
    # 通过增大 RPN 候选框数量来模拟 Fast R-CNN 对更多候选框的处理
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        rpn_pre_nms_top_n_train=2000  # 增大候选框数量
    )
```

------

### 3.3 Faster R-CNN（2015）

#### 核心创新：RPN

Fast R-CNN 仍依赖外部的 Selective Search 生成候选框（约 2 秒/图），成为新的速度瓶颈。Faster R-CNN 引入 **RPN（Region Proposal Network）**，用神经网络直接预测候选框，实现真正的端到端训练。

#### 完整架构

```
输入图像
   ↓
骨干网络（ResNet50 + FPN）→ 多尺度特征图 {P2, P3, P4, P5, P6}
   ↓                              ↓
RPN（区域建议网络）          ← 共享特征图
   ├─ 3×3 卷积扫描特征图
   ├─ 每个位置预测 k=9 个 Anchor（3尺度×3宽高比）
   ├─ 分类分支：前景/背景二分类（objectness score）
   └─ 回归分支：Anchor → 候选框的坐标偏移量
   ↓
NMS 过滤 → Top-N 候选框（训练 2000，测试 300）
   ↓
ROI Align（在特征图上提取候选框特征）
   ↓
检测头：全连接层
   ├─ 分类：Softmax（N+1 类）
   └─ 回归：精修边界框坐标
```

#### Anchor 机制

在特征图的每个位置，预先定义 k 个不同尺度和宽高比的参考框（Anchor）：

- 尺度：128², 256², 512²（像素）
- 宽高比：1:1, 1:2, 2:1

共 3×3=9 个 Anchor，对于 W×H 的特征图，共产生 W×H×9 个候选区域。

#### RPN 损失

$$
L_{RPN} = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)
$$

其中 $p_i^*=1$ 表示该 Anchor 为正样本（与某真值框 IoU > 0.7）。

#### FPN（Feature Pyramid Network）多尺度特征

本实验使用的 `fasterrcnn_resnet50_fpn` 包含 FPN 结构，解决了不同尺度目标的检测难题：

```
ResNet50 自底向上提取特征：
C2(stride=4) → C3(stride=8) → C4(stride=16) → C5(stride=32)

FPN 自顶向下融合：
P5 ← C5
P4 ← C4 + 上采样(P5)
P3 ← C3 + 上采样(P4)
P2 ← C2 + 上采样(P3)
```

小目标在高分辨率特征图（P2/P3）上检测，大目标在低分辨率特征图（P4/P5）上检测。

#### 本实验核心代码解析（`train_RCNN.py`）

**① 数据集类 `UniversalRCNNDataset`**

该类统一处理 VOC（XML）和 COCO（JSON）两种格式：

```python
class UniversalRCNNDataset(Dataset):
    def __init__(self, root, is_coco=False, transforms=None):
        self.class_map = {
            'person': 1, 'bicycle': 2, 'car': 3, 'bus': 4,
            'motorcycle': 5, 'motorbike': 5  # VOC 中叫 motorbike，统一映射到 5
        }

        if is_coco:
            # 预处理：构建 image_id → annotations 的映射字典，避免 O(n²) 查询
            self.ann_map = {int(img['id']): [] for img in data['images']}
            for ann in data['annotations']:
                self.ann_map[int(ann['image_id'])].append(ann)
        else:
            # VOC：直接列出图片文件，标注按文件名对应查找
            self.imgs = [f for f in sorted(os.listdir(self.img_dir))
                         if f.endswith(('.jpg', '.png'))]
```

**② COCO 坐标格式转换**

COCO 标注格式为 `[x, y, w, h]`，需转换为 `[x1, y1, x2, y2]`：

```python
# COCO bbox: [x, y, w, h] → [x1, y1, x2, y2]
x, y, w, h = ann['bbox']
if w > 1 and h > 1:  # 过滤无效标注（宽高为0或极小的框）
    boxes.append([x, y, x + w, y + h])
    labels.append(self.class_map[name])
```

**③ 空标注处理**

部分图片在筛选类别后可能没有任何目标，需返回空 tensor 而非报错：

```python
if len(boxes) == 0:
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "image_id": torch.tensor([idx])
    }
```

**④ 训练循环中的有效样本过滤**

```python
for imgs, tars in pbar:
    valid_imgs, valid_tars = [], []
    for im, ta in zip(imgs, tars):
        # 只处理有标注目标的图片，避免空 batch 导致 loss 计算出错
        if ta['labels'].shape[0] > 0:
            valid_imgs.append(im.to(device))
            valid_tars.append({k: v.to(device) for k, v in ta.items()})

    if not valid_imgs:
        continue  # 整个 batch 都没有目标时跳过
```

**⑤ 模型替换分类头**

预训练模型默认输出 91 类（COCO），需替换为本实验的 6 类（含背景）：

```python
# 获取原始分类头的输入特征维度（通常为 1024）
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用新的 6 类分类头替换
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
```

**⑥ 最优权重保存**

```python
avg_loss = epoch_loss / len(loader)
torch.save(model.state_dict(), f"{out_dir}/last.pth")  # 最新权重
if avg_loss < best_loss:
    best_loss = avg_loss
    torch.save(model.state_dict(), f"{out_dir}/best.pth")  # 最优权重
```

**⑦ 归一化混淆矩阵生成**

```python
cm = confusion_matrix(all_t, all_p, labels=range(6))
# 按行归一化：每行除以该行总和，得到每类的召回率分布
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
# +1e-6 防止某类完全没有真值时出现除零错误
```

------

### 3.4 YOLOv8（You Only Look Once v8，2023）

#### YOLO 系列演进

| 版本   | 年份 | 核心改进                               |
| ------ | ---- | -------------------------------------- |
| YOLOv1 | 2016 | 单阶段检测开山之作，将检测视为回归问题 |
| YOLOv3 | 2018 | 多尺度预测，引入 Darknet-53 骨干       |
| YOLOv5 | 2020 | CSP 结构，工程化完善                   |
| YOLOv8 | 2023 | Anchor-free，解耦检测头，C2f 模块      |

#### YOLOv8 核心架构

```
输入图像 (640×640)
   ↓
Backbone: CSPDarknet + C2f 模块
   → 提取多尺度特征 [80×80, 40×40, 20×20]
   ↓
Neck: PANet（路径聚合网络）
   → 自顶向下 + 自底向上双向特征融合
   ↓
解耦检测头（Decoupled Head）
   ├─ 分类分支：Conv → 类别概率
   └─ 回归分支：Conv → 边界框坐标（DFL Loss）
```

#### Anchor-free 设计

YOLOv8 放弃了预定义 Anchor，改为直接预测目标中心点相对于网格点的偏移量，以及目标的宽高：

- 每个网格点预测：$(x_{offset}, y_{offset}, w, h, cls_1, ..., cls_n)$
- 消除了 Anchor 超参数调优的需求，对不规则形状目标更友好

#### TaskAligned Assigner（正负样本分配）

YOLOv8 使用基于任务对齐的动态样本分配策略，综合考虑分类分数和定位精度：
$$
\text{AlignScore} = s^\alpha \cdot u^\beta
$$
其中 $s$ 为分类得分，$u$ 为 IoU 得分，$\alpha$、$\beta$ 为权重系数。该策略使分类与定位两个任务目标对齐，选出质量最高的正样本。

#### YOLOv8 损失函数

$$
L_{total} = \lambda_1 L_{cls} + \lambda_2 L_{box} + \lambda_3 L_{dfl}
$$

- $L_{cls}$：Binary Cross Entropy（解耦分类头，每类独立二分类）
- $L_{box}$：CIoU Loss（考虑中心距离、宽高比的 IoU 变体）
- $L_{dfl}$：Distribution Focal Loss（将坐标预测视为分布，提升定位精度）

**CIoU Loss 公式**：
$$
L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$
其中 $\rho$ 为中心点距离，$c$ 为最小外接框对角线，$v$ 惩罚宽高比不一致。

#### 本实验核心代码解析（`train_yolo.py`）

```python
def run_compare_experiments():
    experiments = {
        "YOLO_VOC_Balanced": "configs/voc_train.yaml",
        "YOLO_COCO_Balanced": "configs/coco_train.yaml"
    }

    for exp_name, cfg_path in experiments.items():
        model = YOLO("yolov8s.pt")  # 加载 small 预训练权重

        model.train(
            data=cfg_path,      # YAML 配置文件：指定 train/val 路径和类别名
            epochs=30,
            imgsz=640,          # 统一缩放至 640×640 输入
            amp=False,          # 关闭混合精度（部分 GPU 不稳定）
            batch=16,
            workers=1,          # 数据加载进程数（Windows 下建议为 1）
            name=exp_name,
            device=0,           # 使用第 0 块 GPU
            project="runs/train_results"
        )
```

**YAML 配置文件结构**（以 `coco_train.yaml` 为例）：

```yaml
path: dataset/coco_split_mini
train: train/images
val: val/images

nc: 5  # 类别数
names: ['person', 'bicycle', 'car', 'bus', 'motorcycle']
```

**YOLO 标签格式**（txt，每行对应一个目标）：

```
# <class_id> <x_center> <y_center> <width> <height>  （均归一化到 [0,1]）
0 0.512 0.423 0.134 0.287   # person
2 0.712 0.631 0.256 0.312   # car
```

> 这与 VOC/COCO 格式完全不同，需要专门的格式转换脚本，VOC 测试集转换过程中可能出现空 txt 文件。

------

### 3.5 非极大值抑制（NMS）

NMS 是所有目标检测模型后处理的核心，用于消除同一目标的重复检测框：

```python
# NMS 伪代码
def nms(boxes, scores, iou_threshold=0.5):
    # 按置信度降序排列
    sorted_indices = argsort(scores, descending=True)
    keep = []
    while sorted_indices:
        best = sorted_indices[0]   # 取置信度最高的框
        keep.append(best)
        # 计算剩余框与最优框的 IoU
        ious = compute_iou(boxes[best], boxes[sorted_indices[1:]])
        # 保留 IoU 低于阈值的框（与最优框不重叠的框）
        sorted_indices = sorted_indices[1:][ious < iou_threshold]
    return keep
```

Faster R-CNN 训练不充分时，分类置信度分布混乱，导致 NMS 无法有效区分高质量框与低质量重复框，从而出现实验结果中"同一辆自行车三个框"的问题。

------

## 四、训练配置详解

### 4.1 RCNN 系列配置（`train_RCNN.py`）

| 参数         | 值     | 说明                              |
| ------------ | ------ | --------------------------------- |
| 优化器       | SGD    | 动量 SGD，目标检测经典选择        |
| 学习率       | 0.004  | 微调场景，比从头训练（0.02）低    |
| Momentum     | 0.9    | 抑制震荡，加速收敛                |
| Weight Decay | 0.0005 | L2 正则化，防止过拟合             |
| Batch Size   | 1      | 内存限制，RCNN 系列图片尺寸不固定 |
| Epochs       | **5**  | 时间受限，未达收敛                |

**三种模型的差异化配置**：

```python
if model_type == "FasterRCNN":
    # 标准 Faster R-CNN：加载预训练权重
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

elif model_type == "FastRCNN":
    # 模拟 Fast R-CNN：加载预训练权重，增大候选框数量
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        rpn_pre_nms_top_n_train=2000
    )

else:  # RCNN
    # 模拟 R-CNN：不加载预训练权重，从随机初始化训练
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=6)
```

### 4.2 YOLOv8 配置（`train_yolo.py`）

| 参数       | 值         | 说明                               |
| ---------- | ---------- | ---------------------------------- |
| 图像尺寸   | 640        | YOLOv8 标准输入尺寸                |
| Batch Size | 16         | GPU 显存允许范围内尽量大           |
| Epochs     | **30**     | 相对充分，基本看到收敛趋势         |
| AMP        | False      | 关闭自动混合精度，避免训练不稳定   |
| Workers    | 1          | Windows 系统下多进程加载有兼容问题 |
| 预训练权重 | yolov8s.pt | Small 版本，精度与速度均衡         |

------

## 五、推理测试框架：`test.py`

`test.py` 设计了一个统一推理接口 `UniversalDetector`，同时支持 YOLO 与 RCNN 系列模型，支持图片和视频输入。

### 5.1 类初始化与模型加载

```python
class UniversalDetector:
    def __init__(self, model_path, model_type='fasterrcnn', device=None):
        # 统一路径处理：兼容 Windows 反斜杠
        self.model_path = os.path.abspath(model_path.replace('\\', '/'))

        if 'YOLO' in self.model_type:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)  # YOLO 自管理加载流程
        else:
            # RCNN：手动加载权重到指定设备
            self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=6)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()  # 切换到推理模式（关闭 BN 和 Dropout）
```

### 5.2 单帧推理逻辑

```python
def predict_frame(self, frame, conf_threshold=0.5):
    if 'YOLO' in self.model_type:
        # YOLO 推理：直接返回绘制好的图像
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        return results.plot()
    else:
        # RCNN 推理：需手动转换格式并绘制
        img_tensor = F.to_tensor(frame).to(self.device)  # HWC uint8 → CHW float [0,1]
        with torch.no_grad():  # 关闭梯度计算，节省显存
            prediction = self.model([img_tensor])[0]

        # 遍历预测结果，过滤低置信度框
        for box, label, score in zip(
            prediction['boxes'], prediction['labels'], prediction['scores']
        ):
            if score > conf_threshold:
                b = box.cpu().numpy().astype(int)
                cv2.rectangle(res_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                label_text = f"{self.classes[label]}: {score:.2f}"
                cv2.putText(res_img, label_text, (b[0], b[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### 5.3 智能输出文件命名

```python
def _get_save_name(self, source_path):
    # 从权重路径中自动提取数据集信息
    # 例：runs/rcnn_results/FasterRCNN_COCO/best.pth → 提取 "FasterRCNN_COCO"
    for part in reversed(Path(self.raw_model_path).parts):
        if "VOC" in part.upper() or "COCO" in part.upper():
            dataset_info = part
            break
    # 输出文件名格式：原文件名_模型类型_数据集.扩展名
    # 例：img_3_FASTERRCNN_FasterRCNN_COCO.png
    return f"{source_stem}_{self.model_type}_{dataset_info}{source_ext}"
```

这也正是所有结果图片文件名的命名规则来源（如 `img_1_FASTERRCNN_FasterRCNN_COCO.png`）。

### 5.4 视频处理流程

```python
# 视频逐帧处理
cap = cv2.VideoCapture(source_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

with tqdm(total=total_frames, desc="处理进度") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        res_frame = self.predict_frame(frame)  # 对每一帧推理
        out.write(res_frame)
        pbar.update(1)

cap.release()
out.release()
```

------

## 六、检测结果展示与对比分析

### 6.1 场景一：街道人 + 自行车（Faster R-CNN vs YOLOv8s）

同一张输入图，分别用 Faster R-CNN（COCO）和 YOLOv8s（COCO）检测。

**Faster R-CNN 结果（绿色框）：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_1_FASTERRCNN_FasterRCNN_COCO.png" alt="img_1_FasterRCNN" style="zoom: 80%;" />

**YOLOv8s 结果（蓝色框）：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_1_YOLO_YOLO_COCO_Balanced-2.png" alt="img_1_YOLO" style="zoom:80%;" />

**对比分析：**

| 对比项       | Faster R-CNN                     | YOLOv8s                |
| ------------ | -------------------------------- | ---------------------- |
| 检测框数量   | 多，存在明显重复框               | 少，框更精简           |
| bicycle 检测 | 3 个重叠框（0.97 / 0.82 / 0.57） | 1 个框（0.90），无重复 |
| person 检测  | 成功检出（0.60）                 | 成功检出（0.74）       |
| car 检测     | 成功检出（0.75）                 | **未检出**             |
| 框的精准度   | 偏大，有重叠                     | 紧致，贴合目标         |

> **分析**：Faster R-CNN 仅训练 5 epochs，分类置信度分布不稳定，NMS 无法有效抑制低质量重复框（参考 3.5 节 NMS 原理）。同一辆自行车出现 3 个框，本质是模型对该区域多次产生高于阈值的候选框，但分类得分差异不够大，NMS 未能合并。YOLOv8 经过 30 epochs 充分训练，Anchor-free 设计加上 TaskAligned Assigner 使正样本分配更精准，框质量更稳定。

------

### 6.2 场景二：路口复杂多目标（Faster R-CNN vs YOLOv8s）

**Faster R-CNN 结果（绿色框）：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_3_FASTERRCNN_FasterRCNN_COCO.png" alt="img_3_FasterRCNN" style="zoom: 50%;" />

**YOLOv8s 结果（蓝色框）：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_3_YOLO_YOLO_COCO_Balanced-2.png" alt="img_3_YOLO" style="zoom: 50%;" />

**对比分析：**

| 对比项     | Faster R-CNN                 | YOLOv8s                                     |
| ---------- | ---------------------------- | ------------------------------------------- |
| 类别覆盖   | person、car、bicycle         | person、car、bicycle                        |
| 置信度水平 | person: 0.87，car: 0.95~0.97 | person: 0.88~0.93，car: 0.74，bicycle: 0.93 |
| 标签可读性 | 标签位置重叠，可读性差       | 标签清晰，排布合理                          |
| 远处小目标 | 有遗漏                       | bicycle（远处）0.93，召回更好               |
| 框的质量   | 部分框偏大，存在堆叠         | 框更贴合目标轮廓                            |

> **分析**：YOLOv8 的 FPN+PANet 双向特征融合使小目标（远处 bicycle）在高分辨率特征图上得到充分响应，召回更好。Faster R-CNN 虽然对 car 置信度较高，但多目标场景下标签重叠、框堆叠问题突出，说明在训练充分性不足时两阶段模型的后处理更难调优。

------

### 6.3 场景三：俯视路口密集车辆（YOLOv8s）

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_2_YOLO_YOLO_COCO_Balanced-2.png" alt="img_2_YOLO" style="zoom: 67%;" />

**分析：**

- 对多辆 **car** 检测效果出色，置信度 0.85~0.94，俯视视角下框的位置准确
- **person**（0.80）成功检出，模型对俯视人体特征有一定泛化能力
- 右上角摩托车被错误识别为 **car**（0.74 / 0.57）：motorcycle 训练样本远少于 car，且两类在俯视角度下外观差异更小，模型偏向高频类别 car 做出预测——这是训练集类别不均衡的直接体现

------

### 6.4 场景四：公交车 + 密集人群（YOLOv8s）

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\img_YOLO_YOLO_COCO_Balanced-2.png" alt="img_YOLO"  />

**分析：**

- 多个 **person** 均被检出，置信度 0.57~0.82，随遮挡程度增大而降低
- **bus** 成功检出（0.69），大型车辆轮廓明显，有一定识别能力
- 部分遮挡严重的人员置信度低至 0.57 附近，若进一步降低阈值可提升召回但会增加误检

------

### 6.5 Faster R-CNN 独立测试结果

**公交 + 人群场景：**

![res_img](D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\res_img.png)

**街道人 + 自行车场景：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\res_img_1.png" alt="res_img_1" style="zoom:80%;" />

**路口多目标场景：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\res_img_2.png" alt="res_img_2" style="zoom:80%;" />

**俯视路口多车场景：**

<img src="D:\000-MyFile_in_University\硕士\课程相关\深度学习\object_detect\test_results\res_img_3.png" alt="res_img_3" style="zoom: 50%;" />

**分析：**

- `res_img_1`（低密度场景）：Faster R-CNN 正确检出 car（0.66）、person（0.78）、bicycle（0.83），框准确，无明显重复。低密度场景下 RPN 候选框竞争压力小，NMS 效果较好
- `res_img_2`（路口多目标）：标签文字重叠，框偏大，多目标密集时模型表现明显下滑
- `res_img_3`（俯视多车）：car 检测置信度高（0.87~0.94），但 motorcycle 被误分为 car，与 YOLOv8 出现同样问题，说明这是数据侧原因而非模型侧原因
- `res_img`（公交人群）：bus 难以检出，person 相对稳定，印证了样本不均衡对类别精度的直接影响

------

## 七、综合对比分析

### 7.1 Faster R-CNN vs YOLOv8 检测质量对比

| 评估维度   | Faster R-CNN（5 epochs） | YOLOv8s（30 epochs） |
| ---------- | ------------------------ | -------------------- |
| 重复检测   | 较严重（同目标多框）     | 基本无重复           |
| 框的精准度 | 一般，框偏大             | 较好，框贴合目标     |
| 置信度分布 | 跨度大（0.5~0.97）       | 集中（0.7~0.93）     |
| 小目标检测 | 较弱                     | 较强（PANet 加持）   |
| 密集场景   | 易出现框堆叠             | NMS 抑制效果更好     |
| 推理速度   | 慢（两阶段）             | 快（单阶段）         |
| 训练效率   | 低（batch=1）            | 高（batch=16）       |

### 7.2 各类别检测难度分析

| 类别       | 检测难度 | 原因分析                                 |
| ---------- | -------- | ---------------------------------------- |
| person     | 低       | 训练样本最多，特征明显，两模型均稳定检出 |
| car        | 低~中    | 目标大，俯视下两模型均有较好表现         |
| bicycle    | 中       | 与 motorcycle 外形相近，训练充分时可区分 |
| bus        | 中~高    | 训练样本少，大尺寸框容易位置偏移         |
| motorcycle | 高       | 样本最少，频繁被误判为 bicycle 或 car    |

### 7.3 模型架构演进总结

```
R-CNN (2014)
  流程：Selective Search → 逐框 CNN → SVM
  瓶颈：2000 次重复 CNN 计算，速度 47s/图
     ↓ 共享特征图
Fast R-CNN (2015)
  改进：整图特征 + ROI Pooling + 多任务损失
  瓶颈：Selective Search 仍需 2s/图
     ↓ RPN 替代 Selective Search
Faster R-CNN (2015)
  改进：RPN 端到端，速度 0.2s/图，精度大幅提升
  瓶颈：两阶段流程，工程复杂
     ↓ 单阶段直接预测
YOLOv8 (2023)
  改进：Anchor-free，解耦头，速度最快，精度可媲美两阶段
```

------

## 八、问题与反思

1. **训练轮次不足**：RCNN 系列仅 5 epochs 离收敛差距较大，重复检测问题显著；Faster R-CNN 在 20~50 epochs 后 NMS 效果会明显改善
2. **数据类别不均衡**：person 样本量远超其他类别，模型学习时对高频类别过度拟合，混淆矩阵中 motorcycle、bus 的 recall 明显偏低
3. **RCNN 模拟局限**：R-CNN 和 Fast R-CNN 通过参数调整 Faster R-CNN 来模拟，未真正实现 Selective Search，底层算法逻辑存在本质差异，对比结论有一定局限性
4. **置信度阈值未调优**：测试时固定阈值 0.5，不同模型和场景的最优阈值不同，统一阈值可能导致部分模型召回偏低
5. **标签格式转换问题**：VOC 测试集标签在 XML→txt 转换过程中出现空文件，影响了 YOLO 在 VOC 上的部分验证指标
6. **视角泛化问题**：训练集以平视图片为主，俯视角度下模型（尤其对 motorcycle）的泛化能力较弱，需要数据增强或多角度样本补充

------

## 九、结论

本实验完整实现了从数据预处理、模型训练到推理测试的目标检测全流程，主要结论如下：

- **YOLOv8s** 在检测框质量、推理速度和多目标场景的综合表现上优于训练轮次有限的 Faster R-CNN，在小目标和密集场景中优势尤为突出；Anchor-free 设计和 TaskAligned 样本分配使其训练更稳定
- **Faster R-CNN** 在单目标低密度场景下也能达到较好效果，重复检测问题本质是训练不充分导致的置信度分布混乱，而非架构缺陷；充分训练后两阶段模型精度上限更高
- **motorcycle 和 bus** 是所有模型的共同薄弱点，根本原因是训练集严重不均衡，而非模型能力不足——这一点在俯视场景中两个模型同样将 motorcycle 误判为 car 得到印证
- **算法演进脉络**清晰：R-CNN→Fast R-CNN→Faster R-CNN→YOLO 的每一步都是对上一步核心瓶颈的针对性解决，理解这条脉络比单纯比较性能指标更有价值
- **工程实践层面**，YOLOv8 的 `ultralytics` 库封装完善，训练效率（batch=16）远高于手写的 RCNN 训练循环（batch=1），在资源受限场景下是更实用的选择

