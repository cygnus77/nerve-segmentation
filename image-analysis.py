import os
from glob import glob
import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import threading
import concurrent.futures as futures
import itertools
import argparse
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser(description='Dataset analysis tool')
parser.add_argument('-diffs', action='store_true', help='calculate image differences')
parser.add_argument('-hist', action='store_true', help='show histogram of difference amounts')
parser.add_argument('-bucket', action='store', help='set bucket size', default=10000 )
parser.add_argument('-bucketmax', action='store', help='max bucket', default=1e6)
parser.add_argument('-view', nargs='?', action='store', const=100, help='show identical images and masks')

args = parser.parse_args()

if args.diffs:
    filelist = [x for x in glob('./data/train_orig/*.tif') if not '_mask' in x]

    print('num files: %d' % len(filelist))

    image_names = [p[p.rfind('/')+1:-4] for p in filelist]
    # load image data into a 3d numpy array
    image_data = np.array([cv2.imread(p)[:,:,0] for p in filelist])

    batch_size = 200
    print('writing csv')
    with open('./imageset.csv', 'w') as csv:
        csv.write('img1,img2,diff\n')
        for j in range(len(filelist)):
            print(j)
            # load image into cuda as tensor
            img = torch.from_numpy(image_data[j,:,:]).float().cuda()
            img = torch.unsqueeze(img,0)

            # process batches
            for i in range(j+1,len(filelist),batch_size):
                try:
                    # load batch of images into cuda
                    batch = torch.from_numpy(image_data[i:i+batch_size,:,:]).float().cuda()
                    
                    # calculate differences
                    diff = torch.sum(torch.abs(batch - img), [1,2])

                    # write out results
                    for k in range(diff.shape[0]):
                        csv.write('%s,%s,%d\n' % (image_names[j], image_names[i+k], diff[k]))
                except:
                    print(i,j)
                    raise

        csv.close()

if args.hist:
    histo = {}
    count = 0
    with open('imageset2.csv', 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            try:
                diff = int(row[2])
                #print('%s\t%s\t%d' % (row[0], row[1], diff))
                bucket = diff // args.bucket

                if bucket < args.bucketmax:
                    if bucket in histo:
                        histo[bucket] += 1
                    else:
                        histo[bucket] = 1

                    count += 1
            except:
                pass
        f.close()

    buckets = list(histo.keys())
    buckets.sort()

    values = [histo[x] for x in buckets]

    plt.figure()
    plt.bar( buckets, values )
    plt.show()

if args.view is not None:
    with open('imageset2.csv', 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            try:
                diff = int(row[2])
                if diff < args.view:
                    img1 = cv2.imread('./data/train_orig/%s.tif' % row[0])
                    img2 = cv2.imread('./data/train_orig/%s.tif' % row[1])
                    img1_mask = cv2.imread('./data/train_orig/%s_mask.tif' % row[0])
                    img2_mask = cv2.imread('./data/train_orig/%s_mask.tif' % row[1])

                    plt.figure()
                    plt.subplot(2,2,1)
                    plt.imshow(img1, cmap='gray')
                    plt.subplot(2,2,2)
                    plt.imshow(img2, cmap='gray')
                    plt.subplot(2,2,3)
                    plt.imshow(img1_mask, cmap='gray')
                    plt.subplot(2,2,4)
                    plt.imshow(img2_mask, cmap='gray')
                    plt.show()
            except:
                pass
