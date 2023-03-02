import time
import numpy as np
import src.eventdenoisor as edn

from evtool.dvs import DvsFile
from evtool.utils import Player

# nThres, sigmaT, sigmaS
data = DvsFile.load('./data/demo/aedat/s01_v01_c001.aedat4')
model = edn.reclusive_event_denoisor(260, 346, (1.2, 1., 0.7))

# st = time.time()
idx = data['events'].hotpixel(data['size'], thres=1000)
data['events'] = data['events'][idx]

idx = model.run(data['events'].ts, data['events'].x, data['events'].y, data['events'].p)
data['events'] = data['events'][idx]

# print('数据长度: ', (data['events'].ts[-1] - data['events'].ts[0]) * 1E-6, 's')
# print('算法处理用时: ', (time.time() - st), 's')
# 177 193

player = Player(data, core='matplotlib')
player.view(use_aps=True)
