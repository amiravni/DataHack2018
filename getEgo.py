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
  h, w , layers = im1.shape
  im1_tmp = im1[int(h*0):int(h*0.75),:,:]
  im2_tmp = im2[int(h*0):int(h*0.75),:,:]

  im1 = im1_tmp
  im2 = im2_tmp
  
  im1Gray = (cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
  im2Gray = (cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
   
  # Detect ORB features and compute descriptors.
  
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  
  #sift = cv2.xfeatures2d.SIFT_create()
  #keypoints1, descriptors1 = sift.detectAndCompute(im1Gray,None)
  #keypoints2, descriptors2 = sift.detectAndCompute(im2Gray,None)
  
  '''
  print(len(keypoints2))  
  goodIdxs = []
  for iii in range(0,len(keypoints1)):
      if keypoints1[iii].pt[0] > h*0.25 and  keypoints1[iii].pt[0] < h*0.75 and \
           keypoints2[iii].pt[0] > h*0.25 and  keypoints2[iii].pt[0] < h*0.75:
             goodIdxs.append(iii)  
  keypoints1 = list(np.array(keypoints1)[goodIdxs])
  keypoints2 = list(np.array(keypoints2)[goodIdxs])
  descriptors1 = (np.array(descriptors1)[goodIdxs])
  descriptors2 = (np.array(descriptors2)[goodIdxs])  
  '''
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

  
  #points1 = np.float32(points1)
  #points2 = np.float32(points2)
  #print(points1.shape)
  '''
  rows,cols,ch = im1.shape
  print(len(points1),len(points2))
  h = cv2.getAffineTransform(points1,points2)

  im1Reg = cv2.warpAffine(im1,h,(cols,rows))  
  '''
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  
  # Use homography
  height, width, channels = im1.shape
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
            img[idxEl[iii],idxAz[iii]] += pc[iii,3]#rng[iii] * pc[iii,3]#255-((rng[iii]/100)*255).astype('uint8')
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
    min_idx = 31
    decimate = 1
    #elMin = 9999
    #azMin = 9999
    #elMax = -9999
    #azMax = -9999
    ego_last = 0
    img_last = np.zeros((EL_NUM,AZ_NUM,3)).astype('uint8')
    #fgbg = cv2.createBackgroundSubtractorMOG2()

    for idx, frame in enumerate(data_utils.enumerate_frames(video_dir)):
        if idx < min_idx or idx % decimate != 0:
            continue
        pc, ego, label,label_pred,fn = data_utils.read_all_data(video_dir, frame,(0.0,0.0,0.0))
        if idx == min_idx:
            pc_last = pc.copy()
        img_dil_new = pc2img(pc)
        img_dil_old = pc2img(pc_last)
        pc_last = pc.copy()
        
        if idx == min_idx:
            keyFrame = img_dil_old.copy()
        
        
       
        if idx > 0:
            imReg, h = alignImages(img_dil_new,keyFrame)
            rows,cols,ch = img_dil_new.shape
            imReg = cv2.warpPerspective(img_dil_new,h,(cols,rows))
            print(h,ego)
            imgDiff = (np.abs(imReg.astype('float') - keyFrame.astype('float')) ).astype('uint8')
            #print(imgDiff.max())
            #imgDiff = np.zeros((img_dil_new.shape[0],img_dil_new.shape[1],3)).astype('uint8')
            #imgDiff[:,:,0] = imReg[:,:,1]
            #imgDiff[:,:,1] = img_dil_new[:,:,1]
            #imgDiff = np.abs(img_dil_new.astype('float') - img_dil_old.astype('float'))      
            #imgDiff = imgDiff/imgDiff.max()
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
        plt.pause(1.001)
        if idx > 45:
            break
        
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