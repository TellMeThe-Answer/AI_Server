import os
import json
import random
from pathlib import Path
from PIL import Image
from shutil import copy2

names = [
    '00', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
    'a10', 'a11', 'a12', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
    'b8', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c9', 'c11', 'c12'
]

# 데이터셋 경로
img_path = Path(os.getcwd() + "/label/image_file")
label_path = Path(os.getcwd() + "/label/image_label")

# 저장할 디렉토리 경로 설정
output_path = os.getcwd()
train_img_dir = Path(output_path + "/label/train_img/images")
train_label_dir = Path(output_path + "/label/train_img/labels")
val_img_dir = Path(output_path+ "/label/val_img/images")
val_label_dir = Path(output_path + "/label/val_img/labels")

# 디렉토리 생성 (하위 디렉토리 포함)
train_img_dir.mkdir(parents=True, exist_ok=True)
train_label_dir.mkdir(parents=True, exist_ok=True)
val_img_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

# 이미지 및 라벨의 리스트 가져오기
all_images = list(img_path.glob("*.jpg"))
all_labels = list(label_path.glob("*.json"))

# 파일 이름이 일치하는 이미지와 라벨만을 선택
valid_images = []
valid_labels = []

for img_file in all_images:
    corresponding_label = label_path / f"{img_file.stem}.json"
    if corresponding_label.exists():
        valid_images.append(img_file)
        valid_labels.append(corresponding_label)

# 일치하지 않는 데이터 삭제
for img_file in all_images:
    if img_file not in valid_images:
        print("삭제")
        img_file.unlink()

for label_file in all_labels:
    if label_file not in valid_labels:
        print("삭제")
        label_file.unlink()

# 데이터셋을 8:2 비율로 분리
split_ratio = 0.8
total_data = len(valid_images)
train_data_length = int(total_data * split_ratio)
indices = list(range(total_data))
random.shuffle(indices)

train_indices = indices[:train_data_length]
val_indices = indices[train_data_length:]

train_images = [valid_images[i] for i in train_indices]
train_labels = [valid_labels[i] for i in train_indices]
val_images = [valid_images[i] for i in val_indices]
val_labels = [valid_labels[i] for i in val_indices]

# 이미지와 라벨을 해당 디렉토리에 저장하는 함수
def save_to_dir(images, labels, img_dir, label_dir):
    for img_file, label_file in zip(images, labels):
        copy2(img_file, img_dir)
        copy2(label_file, label_dir)

save_to_dir(train_images, train_labels, train_img_dir, train_label_dir)
save_to_dir(val_images, val_labels, val_img_dir, val_label_dir)

# YOLO 형식으로 라벨 변환 후 저장하는 함수
def convert_to_yolo_format(annotation, img_width, img_height):
    box = annotation["annotations"]["bbox"][0]
    class_id = names.index(annotation["annotations"]["disease"])
    cx = (box["x"] + box["w"] / 2) / img_width
    cy = (box["y"] + box["h"] / 2) / img_height
    bw = box["w"] / img_width
    bh = box["h"] / img_height
    return f"{class_id} {cx} {cy} {bw} {bh}"

def process_and_save(images, labels, img_dir, label_dir):
    for img_file, lbl_file in zip(images, labels):
        base_name = img_file.stem

        # 라벨 변환 후 저장
        with open(lbl_file, "r") as f:
            annotation = json.load(f)
            img_width = annotation["description"]["width"]
            img_height = annotation["description"]["height"]
            yolo_format = convert_to_yolo_format(annotation, img_width, img_height)

            with open(label_dir / f"{base_name}.txt", "w") as o:
                o.write(yolo_format + "\n")

process_and_save(train_images, train_labels, train_img_dir, train_label_dir)
process_and_save(val_images, val_labels, val_img_dir, val_label_dir)
