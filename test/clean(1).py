#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-10 14:02

import cv2
import os
import numpy as np

src = r'D:\dataset\DeepGlobe_Road_Extraction\clean\4'
# save = r'E:\PyCharmProject\datasets\save'
if __name__ == '__main__':
    if not os.path.exists(src):
        print('path not exists!')
    counter = 0
    for i, filename in enumerate(os.listdir(src)):
        sp = filename.split('_')
        if sp[1] == 'sat.jpg':
            sat = os.path.join(src, filename)
            mask = os.path.join(src, sp[0] + '_mask.png')

            img_sat = cv2.imread(sat, 1)
            img_mask = cv2.imread(mask, 1)

            cv2.namedWindow(sp[0], 0)
            # cv2.resizeWindow(sp[0], 1536, 768)
            cv2.resizeWindow(sp[0], 900, 900)
            cv2.moveWindow(sp[0], 500, 50)

            # cv2.imshow(sp[0], np.hstack([img_sat, img_mask]))
            cv2.imshow(sp[0], cv2.addWeighted(img_sat, 0.7, img_mask, 0.3, 0))

            if cv2.waitKey(0) & 0xff == ord('d'):
                os.remove(os.path.join(src, filename))
                os.remove(os.path.join(src, sp[0] + '_mask.png'))
                counter += 2
                print('file {} deleted'.format(sp[0]))

            cv2.destroyAllWindows()
        print('{} --- {}'.format(i, filename))
    print('done! {} files were deleted'.format(counter))
