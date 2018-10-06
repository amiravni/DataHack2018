# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import os.path as osp
import sys
sys.path.append('.')
from visualizations.vis import pcshow
import numpy as np
from utilities import data_utils
import os
import cv2
from matplotlib import pyplot as plt

from utilities.math_utils import RotationTranslationData


AZ_NUM = 800#400#800
EL_NUM = 600#300#600

azMin = -80#-41.0
azMax = 80#41.0
elMin = -40#-20.0
elMax = 60#30.0

elRes = (elMax - elMin) / EL_NUM
azRes = (azMax - azMin) / AZ_NUM

elCenter = (elMax - elMin) / 2
azCenter = (azMax - azMin) / 2

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

#az_vec = np.linspace(-40.0, 40.0, num=AZ_NUM )
#el_vec = np.linspace(-10.0, 40.0, num=EL_NUM )

#def pc2img(pcSingle):
#    el = np.rad2deg(np.arctan2(pcSingle[2],pcSingle[0]))
#    az = np.rad2deg(np.arctan2(pcSingle[1],pcSingle[0]))
    
    #rng = pcSingle[0]
'''
def fillHoles(img,(x,y)):
    if img[x,y] > 0:
        return img[x,y]
    else img[x,y] = img[x-1:x+2,y[]]
'''    

def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
 # import pdb;pdb.set_trace()
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
  
def pc2img(pc):

        img = np.zeros((EL_NUM,AZ_NUM))
        img1 = np.zeros((EL_NUM,AZ_NUM))
        img3 = np.zeros((EL_NUM,AZ_NUM,3)).astype('uint8')
        el = np.rad2deg(np.arctan2(pc[:,2],pc[:,0]))#np.linalg.norm(pc[:,0:2],axis=1)))
        az = np.rad2deg(np.arctan2(pc[:,1],pc[:,0]))
        rng = np.sqrt(pc[:,0]**2 + pc[:,1]**2 + pc[:,2]**2)
        idxEl = np.round((elMax - el)/elRes).astype('uint32')
        idxAz = np.round((azMax - az)/azRes).astype('uint32')        
        for iii in range(0,len(rng)):
            img[idxEl[iii],idxAz[iii]] += rng[iii] * pc[iii,3]#255-((rng[iii]/100)*255).astype('uint8')
            img1[idxEl[iii],idxAz[iii]] += pc[iii,3]
        img2 = (img/img1)
        #img2[img1==0] = np.nan
      
        img3[:,:,0] = np.clip((255-(img2 / np.nanpercentile(img2,90))*255),0,255).astype('uint8')
        img4 = img1.copy()
        img4[img1==0] = np.nan       
        img3[:,:,1] = np.clip((255-(img4 / np.nanpercentile(img4,90))*255),0,255).astype('uint8')
        #f = plt.figure(1)
        #print(img1.max())
        #img_blur = cv2.blur(img3,(3,3))
        #imgDiff = (img3.astype('float') - img_last.astype('float'))
        #imgDiff =  np.abs(img3)
        
        kernel = np.ones((2,2), np.uint8)

        #img_erosion = cv2.erode(img, kernel, iterations=1)
        img_dilation = cv2.dilate(img3, kernel, iterations=1)
        return img_dilation
        
if __name__ == '__main__':
    base_dir = os.getcwd()
    #video_dir = os.path.join(base_dir, 'data_examples', 'test_video')
    video_dir = os.path.join(base_dir, 'data', 'train', 'vid_1')
    #import pdb;pdb.set_trace()
    frame_num = data_utils.count_frames(video_dir)
    min_idx = 0
    decimate = 1
    #elMin = 9999
    #azMin = 9999
    #elMax = -9999
    #azMax = -9999
    ego_last = 0
    img_last = np.zeros((EL_NUM,AZ_NUM,3)).astype('uint8')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    for idx, frame in enumerate(data_utils.enumerate_frames(video_dir)):
        if idx < min_idx or idx % decimate != 0:
            continue
        pc, ego, label = data_utils.read_all_data(video_dir, frame,(0.0,0.0,0.0))
        if idx == 0:
            #delta_ego = np.zeros(6)
            pc_last = pc.copy()
        #else:
            #delta_ego = ego_last - ego
        #ego_last = ego.copy() 

        #print(delta_ego)
        #ego_rt = RotationTranslationData(vecs=(delta_ego[:3], delta_ego[3:]))
        ego_rt = RotationTranslationData(vecs=(ego[:3], ego[3:]))
        ego_pc = ego_rt.apply_transform(pc[:, :3])
        ego_pc = np.concatenate((ego_pc, pc[:, 3:4]), -1)

        if idx == 0:
            pc_last = ego_pc.copy()
        
        pcOld2New = ego_rt.inverse().apply_transform(pc_last[:, :3])
        pcOld2New = np.concatenate((pcOld2New, pc_last[:, 3:4]), -1)

        pc_last = ego_pc.copy()   
        img_dil_new = pc2img(pc)
        img_dil_old = pc2img(pcOld2New)

        
        
       
        if idx > 0:
            #imReg, h = alignImages(img_dil_new, img_dil_old)
            #imgDiff = np.abs(imReg.astype('float') - img_dil_old.astype('float'))     
            imgDiff = np.abs(img_dil_new.astype('float') - img_dil_old.astype('float'))      
            imgDiff = imgDiff/imgDiff.max()
            #fgmask = fgbg.apply(img_dil_new)

            plt.imshow(imgDiff)
        else:
            plt.figure(1)
            plt.imshow(img_dil_old)
            plt.pause(0.001)
            #plt.figure(2)
            #plt.imshow(img_dil_new)
            #plt.pause(0.001)
            
        #img_last = img_dilation.copy()
        plt.pause(0.001)
        #break
        #cv2.imshow('frame',img)
        #if cv2.waitKey(1) == 999:
        #    exit(0)        
        #elMin = min(el.min(),elMin)
        #elMax = max(el.max(),elMax)
        #azMin = min(az.min(),azMin)
        #azMax = max(az.max(),azMax)        

        #labeled_pc = np.concatenate((pc, label), -1)
        #pcshow(pc, on_screen_text=osp.join(video_dir, str(frame)), max_points=80000)
        #print(idx,elMin,elMax,azMin,azMax)
        #break