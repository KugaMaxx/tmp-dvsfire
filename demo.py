import cv2
import time
import numpy as np
import src.eventdenoisor as edn
import src.eventdetector as edt

from evtool.dvs import DvsFile
from evtool.utils import Player

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# nThres, sigmaT, sigmaS
data = DvsFile.load('./data/demo/aedat/s04_v05_c001.aedat4')
model = edn.reclusive_event_denoisor(data['size'][0], data['size'][1], (1.2, 1., 0.7))

idx = data['events'].hotpixel(data['size'], thres=1000)
data['events'] = data['events'][idx]

st = time.time()
idx = model.run(data['events'].ts, data['events'].x, data['events'].y, data['events'].p)
data['events'] = data['events'][idx]

print('total timestamp: ', (data['events'].ts[-1] - data['events'].ts[0]) * 1E-6, 's')
print('processing time: ', (time.time() - st), 's')
# 177 193

# player = Player(data, core='matplotlib')
# player.view('1s',use_aps=True)

detector = edt.selective_detector(data['size'][0], data['size'][1], 0.7)
for ts, ev in data['events'].slice('25ms'):
    img = ev.project(data['size'])
    img[img > 0] = 255
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    st = time.time()
    contours = detector.process(ev.ts, ev.x, ev.y, ev.p)
    print(time.time() - st)
    # breakpoint()

    for c in contours:
        x, y, w, h = c
        if (w < 5 and h < 5): continue
        color = np.random.randint(0, 255, 3).tolist()
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("result", data['size'][0]*3, data['size'][1]*3) 
    cv2.imshow("result", img)
    cv2.waitKey(100)
