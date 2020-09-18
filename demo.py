import torch

import cv2
import glob
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from PIL import Image

from networks.stvun import STVUN

from utils import *
from option import args




def Demo(model):

    folder_path = args.data_dir
    folder_name = folder_path.split('/')[-1]
    subfolders  = sorted(glob.glob(folder_path + '/*'))

    if not isdir(args.save_dir +'/' + folder_name +'/'):
        mkdir(args.save_dir +'/' + folder_name +'/')
    

    step = 1 # if step is 2, input frames consist of [0, 2, 4, 6, 8 ...]

    for f in range(len(subfolders)):

        cnt = 0
        imgs = sorted(glob.glob(subfolders[f] + '/*'))
        
        # Set save folder            
        base_path = args.save_dir + '/'+folder_name + '/' + subfolders[f].split('/')[-1]
        
        if not isdir(base_path):
            mkdir(base_path)

        for p in range(0, len(imgs)-6*step, step):

            inputs = []
            
            # Read 7 input frames
            for s in range(7):
                img = np.array(Image.open(imgs[p+s*step]))
                timg = numpy_to_tensor(img)
                timg_pad, wp, hp = pad_img(timg, scale_factor)
                inputs.append(timg_pad)

            # Donwsample input frames using DUF_downsample
            dinputs = DUF_downsample(inputs, scale=scale_factor)

            # Set number of intermediate frames
            tList = np.linspace(1, args.time_step+1, args.time_step+1) / (args.time_step + 1)
            tList = np.delete(tList, -1)
            
            # Forward input frames
            outs  = model(dinputs, tList=tList)
            
            # Space upsampling output
            sout = outs[0] 
            sout = clip_and_numpy(sout, wp, hp)

            save_img(sout, base_path + '/Frame' + num_to_filename(cnt) + '.png')
            cnt +=1

            # Space-time upsampling outputs
            for l in range(len(tList)):
                stout = outs[l+1]
                stout = clip_and_numpy(stout, wp, hp)

                save_img(stout, base_path + '/Frame' + num_to_filename(cnt) + '.png' )
                cnt +=1

            

scale_factor = 4 

# load pretrained model
network = STVUN().cuda()
network = load_pretrained_weight(network, args.pre_train)
    
# Demo start
print('===> Demo start')

if not isdir(args.save_dir):
    mkdir(args.save_dir)

with torch.no_grad():
    Demo(network)
