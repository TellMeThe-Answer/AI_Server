
import torch
from torchvision import models
import os
import sys


sys.path.insert(0, './model')


# 모델 로딩
# model = torch.load("./model/best_model.pt")
model = torch.hub.load('./yolov5', 'custom', path='./model/best_model.pt', source='local')

train_img = './256914_20210830_2_1_a6_3_2_12_3_1.jpg'

temp = model(train_img)
# 결과 출력
temp.print()

# 결과 이미지 저장
temp.save()  # save results (image with detections)
