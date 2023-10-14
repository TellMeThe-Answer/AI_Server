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
        
        bounding_boxes = temp.pred[0][:, :4]  # 예측된 바운딩 박스 정보 (x1, y1, x2, y2)
        class_predictions = temp.pred[0][:, 5:]  # 클래스 예측 확률 (클래스에 속하는 각 객체에 대한 확률)
        confidence_scores = temp.pred[0][:, 4]  # 바운딩 박스의 신뢰도 점수

        print(bounding_boxes)
        print(class_predictions)
        print(confidence_scores)

        return "1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)