from flask import Flask, jsonify, request
import torch
from torchvision import models
import os
import sys
import json

app = Flask(__name__)

disease_code = [
    '00', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
    'a10', 'a11', 'a12', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
    'b8'
]

disease_name = [
    '정상', '딸기잿빛곰파이병', '딸기흰가루병', '오이노균병', '오이흰가루병', '토마토흰가루병', '토마토잿빛곰파이병',
    '고추탄저병', '고추흰가루병', '파프리카흰가루병', '파프리카잘록병', '시설포도탄저병', '시설포도노균병',
    '냉해피해', '열과', '칼슘결핍', '일소피해', '축과병', '다량원소결핍 (N)', '다량원소결핍 (P)', '다량원소결핍 (K)'
]

## 모델 옵션

# 객체 탐지 수
# model.max_det = 4

# 신뢰도 값
# model.conf = 0.01 

# 라벨링이 여러개가 가능하도록 할지
# model.multi_label = True  

#  IoU가 높을수록 두 bounding box가 많이 겹치고 있음을 의미하며, IoU가 낮을수록 두 bounding box가 겹치는 정도가 적다는 것을 나타냅니다. 0.4 ~ 0.5 값
# model.iou = 0.45 


sys.path.insert(0, './model')

# 모델 로딩
tomato_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
strawberry_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
cucumber_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
pepper_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')

# 모델 옵션
def set_model_option(model):
    model.max_det = 4
    model.conf = 0.01 
    model.multi_label = True  
    model.iou = 0.45 

# 이미지 저장
def save_image(file):
    file.save('./img/'+ file.filename)

# 결과 code를 한글로 매치
def match_name(code):
    for index in range(len(disease_code)):
        if code == disease_code[index]:
            return (disease_name[index])
    return None

# 작물 따라서 다른 모델 적용
def run_crop_model(crop_type, train_img, img_size):
    if crop_type == 'tomato':
        return tomato_model(train_img, size = img_size)
    elif crop_type == 'strawberry':
        return strawberry_model(train_img, size = img_size)
    elif crop_type == 'cucumber':
        return cucumber_model(train_img, size = img_size)
    elif crop_type == 'pepper':
        return pepper_model(train_img, size = img_size)
    else:
        return None
    
    
@app.route('/')
def hello():
    return 'Hello World!'

# 작물 예측
@app.route('/predict', methods=['POST'])
def predict():
   
    data = {}
     
    # 작물 타입
    crop_type = request.form['type']
    input_img = request.files['image_file']
   
    save_image(input_img) # 들어오는 이미지 저장
    train_img = './img/' + input_img.filename
    
    result = run_crop_model(crop_type, train_img, 416) # 모델 실행
    
    if result == None:
        return jsonify({"contents" : "잘못된 작물입니다.", "result" : False})
    
    ouput = result.pandas().xyxy[0] # 결과 text데이터
    
    result.print() # 결과 출력
    result.save()  # 결과 이미지 저장
    
    
    data['result'] = True
    
    crop_reulst =[]
    for idx in ouput.index:

        name = match_name(ouput.loc[idx, 'name'])
        confidence = round(ouput.loc[idx, 'confidence'], 4)
        crop_reulst.append({"crop" : name, "confidence" : confidence})
    
    data['contents'] = crop_reulst
    
    return jsonify(data)

if __name__ == "__main__":
    
    # 모델 옵션 적용
    set_model_option(tomato_model)
    set_model_option(cucumber_model)
    set_model_option(pepper_model)
    set_model_option(strawberry_model)
    
    
    app.run(host='0.0.0.0', port=8080, debug=True)