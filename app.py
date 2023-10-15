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

# 객체 탐지 수
model.max_det = 4
# 신뢰도 값
model.conf = 0.01 
# 라벨링이 여러개가 가능하도록 할지
model.multi_label = True  
#  IoU가 높을수록 두 bounding box가 많이 겹치고 있음을 의미하며, 
#  IoU가 낮을수록 두 bounding box가 겹치는 정도가 적다는 것을 나타냅니다.
#  일반적으로 NMS IoU threshold는 0.4에서 0.5 사이의 값
model.iou = 0.45  # NMS IoU threshold  



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
        result = model(train_img, size = 640)
        
        # 결과 출력
        result.print()
        # 결과 이미지 저장
        result.save()  # save results (image with detections)

        print(result.pandas().xyxy[0])
        
        return "1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)