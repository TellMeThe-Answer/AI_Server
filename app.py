from flask import Flask, jsonify, request
import torch
from torchvision import models
import os
import sys

app = Flask(__name__)

sys.path.insert(0, './model')


# 모델 로딩
model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')

# 모델 옵션

model.max_det = 3  # 객체 탐지 수

model.conf = 0.99  # 신뢰도 값

    #   iou = 0.45  # NMS IoU threshold
    #   agnostic = False  # NMS class-agnostic
    #   multi_label = False  # NMS multiple labels per box
    #   classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    #   max_det = 1000  # maximum number of detections per image
    #   amp = False  # Automatic Mixed Precision (AMP) inference

# results = model(im, size=320)  # custom inference size

# 이미지 저장
def save_image(file):
    file.save('./img/'+ file.filename)
    

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image_file']
        save_image(file) # 들어오는 이미지 저장
        train_img = './img/' + file.filename
        temp = model(train_img, size=416)
        
        # 결과 출력
        temp.print()
        # 결과 이미지 저장
        temp.save()  # save results (image with detections)
        


        return "1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)