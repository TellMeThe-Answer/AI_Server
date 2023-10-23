from flask import Flask, jsonify, request, send_file
import torch
from torchvision import models
import os
import sys
import json
from datetime import datetime
from flask_restx import Api, Resource, reqparse
from flask_cors import CORS

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

input_dir = './img/input/'
output_dir = './img/output/'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


sys.path.insert(0, './model')

CORS(app, resources={r"/*": {"origins": "*"}})

# api swagger
api = Api(app, version='1.0', title='API 문서', description='Swagger 문서', doc="/api-docs")
predict_api = api.namespace('predict', description='작물병해 판단')

# 모델 로딩
tomato_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
strawberry_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
cucumber_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')
pepper_model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')

# 모델 옵션
def set_model_option(model):
    model.max_det = 4  # 객체 탐지 수
    model.conf = 0.01  # 신뢰도 값
    model.multi_label = True   # 라벨링이 여러개가 가능하도록 할지
    model.iou = 0.45  # 0.4 ~ 0.5 값

# 이미지 저장
def save_image(file):
    file.save(input_dir+ file.filename)

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

# 이미지 고유시간으로 이름변경 
def change_img_name(file):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    changed_name = (timestamp) + file.filename
    os.rename(input_dir+ file.filename, input_dir + changed_name)
    
    return changed_name  

# 판단결과를 list로 리턴
def add_result_list(result):
    ouput = result.pandas().xyxy[0] # 결과 text데이터
    crop_reulst =[]
    
    for idx in ouput.index:
        name = match_name(ouput.loc[idx, 'name'])
        confidence = round(ouput.loc[idx, 'confidence'], 4)
        crop_reulst.append({"crop" : name, "confidence" : confidence})  
        
    return crop_reulst

# 유요한 작물타입인지 확인
def is_valid_crop(crop_type):
    if crop_type == 'tomato' or crop_type == 'strawberry' or crop_type == 'cucumber' or crop_type == 'pepper':
        return True
    else:
        return False

# 허용된 파일 형식인지 확인
def is_allowed_file(input_img):
    return '.' in input_img and input_img.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 업로드된 파일이 정상인지 확인
def is_exist_file(input_img):
    return (str(input_img) == "<FileStorage: '' (None)>" or input_img.filename == '')

@app.route('/home')
def hello():
    return 'Hello World!'

@predict_api.route('/')
class Predict(Resource):
    def post(self):  # 작물 예측
    
        data = {}
        
        crop_type = request.form.get('type')
        input_img = request.files['image_file']
        
        # 작물 타입
        if not crop_type:
            return jsonify({"contents": "작물정보가 업로드 되지 않았습니다.", "result": False})
        
        # 작물 입력 오류
        if not is_valid_crop(crop_type):
            return jsonify({"contents" : "tomato, strawberry, cucumber, pepper 중 하나를 입력해주세요.", "result" : False})
        
        # 파일이 제대로 업로드 되었는지 확인
        if is_exist_file(input_img):
            return jsonify({"contents" : "이미지가 업로드 되지 않았습니다.", "result" : False})

        # 파일 형식이 jpeg, jpg, png가 맞는지
        if not is_allowed_file(input_img.filename):
            return jsonify({"contents" : "jpeg, jpg, png형식의 파일을 업로드해주세요.", "result" : False})
 
        # 이미지 저장, 이름 변경
        save_image(input_img) 
        unique_name = change_img_name(input_img)
        
        # 모델 실행
        train_img = input_dir + unique_name
        result = run_crop_model(crop_type, train_img, 416)
        
        # 모델 결과 이미지
        result.print() 
        result.save(save_dir=output_dir,exist_ok=True)  
        
        # 결과값 리스트로 저장
        crop_reulst = add_result_list(result)
        
        data['result'] = True
        data['contents'] = crop_reulst
        data['image_path'] = unique_name # 결과에 이미지 url
        
        return jsonify(data)
    
    # 결과 이미지 요청
    def get(self):
        data = request.json
        
        try:
            image_path = output_dir + data['image_name'] 
            return send_file(image_path, mimetype='image/jpeg')

        except FileNotFoundError:
            response = jsonify({'error': 'Image not found', 'result' : False})
            response.status_code = 404
            return response
    
if __name__ == "__main__":
    
    # 모델 옵션 적용
    set_model_option(tomato_model)
    set_model_option(cucumber_model)
    set_model_option(pepper_model)
    set_model_option(strawberry_model)
    
    
    app.run(host='0.0.0.0', port=8080, debug=True)

