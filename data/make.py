import os
import os.path as osp

import cv2
import time
import argparse
import numpy as np
from tqdm import trange, tqdm
from evtool.dvs import DvsFile

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def transform_event(count_map):
    image = np.zeros((*count_map.shape, 3))
    image[count_map > 0, 2] = 255
    image[count_map < 0, 1] = 255
    return image.astype(np.uint8)


def transform_frame(aps_frame):
    image = aps_frame
    return image.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="paramters")
    parser.add_argument('-r', '--replace', action='store_false')
    parser.add_argument('-ev', '--events', action='store_false')
    parser.add_argument('-fr', '--frames', action='store_false')
    parser.add_argument('-hy', '--hybrid', action='store_false')
    args = parser.parse_args()
    
    cwd = osp.dirname(__file__)
    output = create(osp.join(cwd, './image_set'))

    fig, cmap = plt.figure(), mcolors.LinearSegmentedColormap.from_list('custom', ['#00FF00', '#000000', '#FF0000'])
    for root, dirs, files in os.walk(osp.join(cwd, './aedat'), topdown=False):
        files.sort()
        for file in tqdm(files):
            name, ext = osp.splitext(file)
            if osp.exists(osp.join(output, name)) and args.replace is False:
                time.sleep(0.1)
                continue

            sub_dir = create(osp.join(output, name))
            sub_dir_ev = create(osp.join(sub_dir, 'events'))
            sub_dir_fr = create(osp.join(sub_dir, 'frames'))
            sub_dir_hb = create(osp.join(sub_dir, 'hybrid'))

            data = DvsFile.load(osp.join(root, file))
            for ev_ts, ev in data['events'].slice('25ms'):
                ev_img, ev_img_name = ev.project(data['size']), '{:16d}.png'.format(ev_ts)
                ev_img = transform_event(ev_img)
                cv2.imwrite(osp.join(sub_dir_ev, ev_img_name), ev_img)

                fr_ts, fr_img = data['frames'].find_closest(ev_ts)
                fr_img = transform_frame(fr_img)

                hy_img = cv2.addWeighted(fr_img, 0.6, ev_img, 0.5, 0)
                cv2.imwrite(osp.join(sub_dir_hb, ev_img_name), hy_img)

            for fr_ts, fr in data['frames']:
                fr, fr_name = transform_frame(fr), '{:16d}.png'.format(fr_ts)
                cv2.imwrite(osp.join(sub_dir_fr, fr_name), fr_img)
