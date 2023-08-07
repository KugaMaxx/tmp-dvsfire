import os
import cv2
import time
import numpy as np

import pandas as pd

from evtool.dvs import DvsFile
from evtool.utils import Player

from modules import kore
from modules import reclusive_event_denoisor as edn
from modules import selective_event_detector as edt

import matplotlib.pyplot as plt


from sklearn.metrics import precision_score, recall_score

def calc_iou(boxA, boxB):
    x1,y1,x2,y2 = boxA #box1的左上角坐标、右下角坐标
    x3,y3,x4,y4 = boxB #box1的左上角坐标、右下角坐标

    #计算交集的坐标
    x_inter1 = max(x1,x3) #union的左上角x
    y_inter1 = max(y1,y3) #union的左上角y
    x_inter2 = min(x2,x4) #union的右下角x
    y_inter2 = min(y2,y4) #union的右下角y

    # 计算交集部分面积，因为图像是像素点，所以计算图像的长度需要加一
    # 比如有两个像素点(0,0)、(1,0)，那么图像的长度是1-0+1=2，而不是1-0=1
    interArea = max(0,x_inter2-x_inter1+1)*max(0,y_inter2-y_inter1+1)

    # 分别计算两个box的面积
    area_box1 = (x2-x1+1)*(y2-y1+1)
    area_box2 = (x4-x3+1)*(y4-y3+1)

    #计算IOU，交集比并集，并集面积=两个矩形框面积和-交集面积
    return interArea/(area_box1+area_box2-interArea)

file_path = '/media/kuga/OS/Users/40441/Desktop/DvFire/data/s4/s04_v03_c002.aedat4'
data = DvsFile.load(file_path)

file_name = os.path.splitext(os.path.split(file_path)[1])[0]
gt = pd.read_csv('/media/kuga/OS/Users/40441/Desktop/DvFire/det/'+f'{file_name}.txt', names=['file_name', 'class', 'id', 'x', 'y', 'w', 'h'])
gt['timestamp'] = np.array([int(f.split('.')[0]) for f in gt['file_name']])

label = pd.DataFrame({'timestamp': np.arange(gt['timestamp'].iloc[0], gt['timestamp'].iloc[-1] + 25000, 25000)})
label['true'] = np.zeros(len(label)).astype(np.bool_)
label['pred'] = np.zeros(len(label)).astype(np.bool_)

ev_noise = data['events'].copy()

# idx = data['events'].hotpixel(data['size'], thres=1000)
# data['events'] = data['events'][idx]

TP, FP, FN = 0, 0, 0
pred_num = 0
detector = edt.init(data['size'][0], data['size'][1])
for i, (ts, ev) in enumerate(data['events'].slice('25ms', from_ts=label['timestamp'].iloc[0])):
    pred_flag = False

    img = ev.project(data['size']).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # label.loc[label['timestamp'] == ts, 'pred'] = True
    detect_rect = detector.run(ev, threshold=0.80, num=2)
    for x, y, w, h in detect_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
        boxA = (x, y, x+w, y+h)
        pred_num += 1
        pred_flag = True

    for k in np.where(ts == gt['timestamp']):
        sub_gt = gt.iloc[k]
        # label.loc[label['timestamp'] == int(sub_gt['timestamp'].iloc[0]), 'true'] = True
        x, y, w, h = int(sub_gt['x'].values), int(sub_gt['y'].values), int(sub_gt['w'].values), int(sub_gt['h'].values)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))
        boxB = (x, y, x+w, y+h)

        if pred_flag is False:
            FN += 1
        elif calc_iou(boxA, boxB) >= 0.50: 
            TP += 1
        else:
            FP += 1
    
    cv2.namedWindow('result', cv2.WINDOW_FREERATIO) 
    cv2.imshow('result', img)
    cv2.waitKey(1)

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print(f"precision:{precision}")
print(f"recall:{recall}")
print(f"F1:{2 * (precision * recall) / (precision + recall)}")


model = edn.init(data['size'][0], data['size'][1]) # nThres, sigmaT, sigmaS
idx = model.run(data['events'], threshold=0.01)
ev_filter = data['events'][idx]

TP, FP, FN = 0, 0, 0
pred_num = 0
detector = edt.init(data['size'][0], data['size'][1])
for i, (ts, ev) in enumerate(ev_filter.slice('25ms', from_ts=label['timestamp'].iloc[0])):
    pred_flag = False

    img = ev.project(data['size']).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # label.loc[label['timestamp'] == ts, 'pred'] = True
    detect_rect = detector.run(ev, threshold=0.80, num=2)
    for x, y, w, h in detect_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
        boxA = (x, y, x+w, y+h)
        pred_num += 1
        pred_flag = True

    for k in np.where(ts == gt['timestamp']):
        sub_gt = gt.iloc[k]
        # label.loc[label['timestamp'] == int(sub_gt['timestamp'].iloc[0]), 'true'] = True
        x, y, w, h = int(sub_gt['x'].values), int(sub_gt['y'].values), int(sub_gt['w'].values), int(sub_gt['h'].values)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))
        boxB = (x, y, x+w, y+h)

        if pred_flag is False:
            FN += 1
        if calc_iou(boxA, boxB) >= 0.50: 
            TP += 1
        else:
            FP += 1
    
    cv2.namedWindow('result', cv2.WINDOW_FREERATIO) 
    cv2.imshow('result', img)
    cv2.waitKey(1)

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print(f"precision:{precision}")
print(f"recall:{recall}")
print(f"F1:{2 * (precision * recall) / (precision + recall)}")
