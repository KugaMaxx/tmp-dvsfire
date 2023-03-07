import cv2
import time
import numpy as np
import eventdenoisor as edn
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
idx = model.run(data['events'])
data['events'] = data['events'][idx]

print(idx.sum())
