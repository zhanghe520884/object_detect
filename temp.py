import os
import json

# --- 配置区：请根据你的实际文件夹结构修改 ---
DATA_PATHS = {
    "VOC_Train": {"path": "dataset/voc_split_mini/train/annotations", "type": "xml"},
    "VOC_Test": {"path": "dataset/voc_split_mini/test/annotations", "type": "xml"},
    "COCO_Train": {"path": "dataset/coco_split_mini/train/annotations", "type": "json"},
    "COCO_Test": {"path": "dataset/coco_split_mini/test/annotations", "type": "json"}
}


def count_images():
    print(f"{'数据集名称':<15} | {'总图片张数':<12} | {'文件类型':<10}")
    print("-" * 45)

    for name, config in DATA_PATHS.items():
        folder_path = config['path']
        if not os.path.exists(folder_path):
            print(f"{name:<15} | {'路径不存在':<12} | {config['type']:<10}")
            continue

        count = 0
        if config['type'] == 'xml':
            # 统计 XML 文件的数量，每个 XML 对应一张图片
            count = len([f for f in os.listdir(folder_path) if f.endswith('.xml')])

        elif config['type'] == 'json':
            # 解析 JSON 文件中的 'images' 列表长度
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if json_files:
                try:
                    with open(os.path.join(folder_path, json_files[0]), 'r') as f:
                        data = json.load(f)
                        count = len(data.get('images', []))
                except Exception as e:
                    print(f"解析 {name} 出错: {e}")

        print(f"{name:<15} | {count:<12} | {config['type']:<10}")


if __name__ == "__main__":
    count_images()