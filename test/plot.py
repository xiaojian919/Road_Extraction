#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-23 17:46


import os
from matplotlib import pyplot as plt
import numpy as np


def get_info(path):
    if not os.path.exists(path):
        print('error: ' + path + ' not exists')
        exit()
    files = os.listdir(path)

    with open(os.path.join(path, files[0])) as f:
        label = f.readline().strip().split('\t')[1:]

    logs = []
    a = {}
    for file in files:
        # a.update({file: np.loadtxt(os.path.join(path, file), skiprows=1)})
        logs.append(np.loadtxt(os.path.join(path, file), skiprows=1))

    epoch = []
    loss = []
    pa = []
    iou = []
    precision = []
    for log in logs:
        epoch.append(log[..., :1])
        loss.append(log[..., 1:2])
        pa.append(log[..., 2:3])
        iou.append(log[..., 3:4])
        precision.append(log[..., 4:])

    return epoch, loss, pa, iou, precision, label, files

def get_list(list1):
    list1 = [i for item in list1 for i in item]
    return list(map(float, list1))

def draw(xs, ys, label, name, loc, save):

    global get_list, max
    plt.figure()
    plt.title('epoch - ' + label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    for i in range(len(name)):
        x = get_list(xs[i])
        y = get_list(ys[i])
        print(name[i].split('24k')[0] + '\t' + str(max(y)))
        # if label != 'loss':
        #     plt.text(110, 0.5 - i * 0.03, str(round(max(y), 3)))
        plt.plot(x, y, label=name[i].split('24k')[0])
    plt.legend(loc=loc)
    plt.savefig(os.path.join(save, label))
    plt.show()


if __name__ == '__main__':
    path = './log'
    save = './eval'
    if not os.path.exists(save):
        os.mkdir(save)
    epoch, loss, pa, iou, precision, label, name = get_info(path)
    draw(epoch, loss, label[0], name, 'upper right', save)
    draw(epoch, pa, label[1], name, 'lower right', save)
    draw(epoch, iou, label[2], name, 'lower right', save)
    draw(epoch, precision, label[3], name, 'lower right', save)
