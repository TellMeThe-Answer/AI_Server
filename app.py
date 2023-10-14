from flask import Flask, jsonify, request
import torch
from torchvision import models
import os
import sys

app = Flask(__name__)

sys.path.insert(0, './model')


# 모델 로딩
model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')

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
        temp = model(train_img)
        
        # 결과 출력
        temp.print()

        # 결과 이미지 저장
        temp.save()  # save results (image with detections)

        return "sdf"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)