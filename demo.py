import cv2
import time
import numpy as np

from evtool.dvs import DvsFile
from evtool.utils import Player

from modules import kore
from modules import event_denoisor as edn
from modules import event_detector as edt

import matplotlib.pyplot as plt


data = DvsFile.load('./data/demo/aedat/s04_v05_c001.aedat4')

# idx = data['events'].hotpixel(data['size'], thres=1000)
# data['events'] = data['events'][idx]

# st = time.time()
model = edn.reclusive_event_denoisor(data['size'][0], data['size'][1]) # nThres, sigmaT, sigmaS
# print(idx.sum())



idx = model.run(data['events'], samplarT=-0.5, sigmaT=5, sigmaS=2)
print(data['events'].shape)
data['events'] = data['events'][idx]
print(idx.sum())

# fig, (ax1, ax2) = plt.subplots(1, 2)

detector = edt.selective_detector(data['size'][0], data['size'][1])
for ts, ev in data['events'].slice('25ms'):
    rects = detector.run(ev, threshold=0.8)

    img = ev.project(data['size']).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
    
    cv2.namedWindow('result', cv2.WINDOW_FREERATIO) 
    cv2.imshow('result', img)
    cv2.waitKey(1)

    # # ev = ev[model.run(ev, threshold=thres)]
    # plt.imshow(ev.project(data['size']), vmin=-1, vmax=1, cmap=plt.set_cmap('bwr'))
    
    # plt.savefig("temp{}.jpg".format(ts))