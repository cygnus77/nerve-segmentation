import cv2
from glob import glob
import os
import numpy as np

filelist = [x for x in glob('./data/train_orig/*.tif') if not '_mask' in x]

print('num files: %d' % len(filelist))

images = [(p[p.rfind('/')+1:-4] ,cv2.imread(p)) for p in filelist]

with open('./imageset.csv', 'w') as csv:
    csv.write('img1,img2,diff\n')
    for i in images:
        print(i[0])
        for j in images:
            diff = np.sum(i[1]-j[1])
            csv.write('%s,%s,%d\n' % (i[0], j[0], diff))
    csv.close()
