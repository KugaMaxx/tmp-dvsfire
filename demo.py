import cv2
import time
import numpy as np

from evtool.dvs import DvsFile
from evtool.utils import Player

from modules import kore
from modules import event_denoisor as edn
from modules import event_detector as edt

import matplotlib.pyplot as plt

# nThres, sigmaT, sigmaS
data = DvsFile.load('./data/demo/aedat/s04_v05_c001.aedat4')

# Access dimensions of the event stream
height, width = data['size']

data['events'] = data['events'][:10000]

projected_events = data['events'].project(data['size'])
plt.imshow(projected_events, vmin=-1, vmax=1, cmap=plt.set_cmap('bwr'))
plt.show()

# model = edn.reclusive_event_denoisor(data['size'][0], data['size'][1], (1.2, 1., 0.7))

# idx = data['events'].hotpixel(data['size'], thres=1000)
# data['events'] = data['events'][idx]

# st = time.time()
# idx = model.run(data['events'])
# data['events'] = data['events'][idx]
# print(idx.sum())

# detector = edt.selective_detector(data['size'][0], data['size'][1], 0.7)
# for ts, ev in data['events'].slice('25ms'):
#     st = time.time()
#     rects = detector.run(ev)
#     print(time.time() - st)

#     img = ev.project(data['size']).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     for x, y, w, h in rects:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
    
#     cv2.namedWindow('result', cv2.WINDOW_FREERATIO) 
#     cv2.imshow('result', img)
#     cv2.waitKey(0)

#     breakpoint()

# img = data['events'].project(data['size'])
# img = img / ((data['events'].timestamp[-1] - data['events'].timestamp[0]) * 1E-6)
# plt.imshow(img, vmin=0, vmax=10)
# plt.show()
