# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.
import os
os.chdir("D:\\DataHack18\\Dataset\\\DataHack2018-master")
import os.path as osp
import sys
sys.path.append('.')
from visualizations.vis import pcshow
import numpy as np
from utilities import data_utils
from utilities.math_utils import RotationTranslationData
import os
import math
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import time
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

    
AZ_NUM = 800#800
EL_NUM = 400#600
maxDist=100


maxDist_x=120
maxDist_y=150

maxDist_x=150
maxDist_y=150


azMin = -40.0
azMax = 40.0
elMin = -10.0
elMax = 40.0

elRes = (elMax - elMin) / EL_NUM
azRes = (azMax - azMin) / AZ_NUM

# 100*math.tan(40*math.pi/180)
pix2m_x=AZ_NUM/maxDist_x
pix2m_y=EL_NUM/maxDist_y

elCenter = (elMax - elMin) / 2
azCenter = (azMax - azMin) / 2

saveFig=False
EGO_Flag=True # use world fixed c.s
diffFalg=True  # use image diff.
showLabel=True # show labels
KeyFrameUpdate=50 # [frames]
writeLabelFlag=False # write new labels
#az_vec = np.linspace(-40.0, 40.0, num=AZ_NUM )
#el_vec = np.linspace(-10.0, 40.0, num=EL_NUM )

#def pc2img(pcSingle):
#    el = np.rad2deg(np.arctan2(pcSingle[2],pcSingle[0]))
#    az = np.rad2deg(np.arctan2(pcSingle[1],pcSingle[0]))
    
    #rng = pcSingle[0]
def pc2im_PerProj(y,z,intens,label,prev_img,idx,diffFalg=False):
    img = np.zeros((EL_NUM+1,AZ_NUM+1))
    imgL = np.zeros((EL_NUM+1,AZ_NUM+1))
    # el = np.rad2deg(np.arctan2(z,x))
    # az = np.rad2deg(np.arctan2(y,x))
    # rng = np.sqrt(x**2 + y**2 + z**2)
    idxEl = np.round(z*EL_NUM).astype('uint32')
    idxAz = np.round(y*AZ_NUM).astype('uint32')        
    for iii in range(0,len(intens)):
        try:
            # img[idxEl[iii],idxAz[iii]] = 255-((rng[iii]/maxDist)*255).astype('uint8')
            img[idxEl[iii],idxAz[iii]] = intens[iii]
            imgL[idxEl[iii],idxAz[iii]] = label[iii]
        except:
            print('out of range')
    
    #f = plt.figure(1)
    # plt.imshow(img)
    # plt.imshow(img-prev_img)
    kernel = np.ones((3,3), np.uint8)

        #img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    if diffFalg:
        plt.title('Frame #{}'.format(str(idx)))
        plt.imshow(img_dilation-prev_img)

    else:
        plt.title('Frame #{}'.format(str(idx)))
        plt.imshow(img_dilation)
        plt.hold
        plt.imshow(imgL)
        if saveFig:
            plt.savefig('fig_'+str(idx)+'.png')
        
        # plt.t
    plt.pause(0.05)
    return img_dilation
    
def genClusSTD(labels,X,clusterNum):
    tmp=np.argwhere(labels == clusterNum)
    Xstd=np.std(X[tmp],axis=0)
    return Xstd
    
def clustMake(centroid,idx,X):
    if idx>0:
        kmeans = KMeans(n_clusters=30,init=centroid, random_state=0).fit(X)
    else:
        kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
        # model = SpectralClustering(n_clusters=35,affinity='nearest_neighbors',assign_labels='kmeans')
    # else:
        
        # model = SpectralClustering(n_clusters=35,init=affinity='nearest_neighbors',assign_labels='kmeans')
    # model = SpectralClustering(affinity='rbf', assign_labels='discretize', coef0=1,
                # degree=3, eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                # kernel_params=None, n_clusters=2, n_init=10, n_jobs=None,
                # n_neighbors=10, random_state=0)
    # labels = model.fit_predict(X)
    kmeans.predict(X)
    labels=kmeans.labels_
    Cluster_std_vec = []
    for i in range(max(labels)):
        Cluster_std_vec.append(genClusSTD(labels,X,i))
    
    centroids=kmeans.cluster_centers_
    # plt.scatter(X[:, 0], X[:, 1], c=labels,s=42, cmap='viridis');
    # plt.xlim(0,400)
    # plt.ylim(0,400)
    # plt.hold(False)
    # time.sleep(1)
    return centroids,Cluster_std_vec

def pc2im(x,y,z,intens,label,prev_img,idx,diffFalg=False):
    img = np.zeros((EL_NUM+1,AZ_NUM+1))
    # imgL = np.zeros((EL_NUM+1,AZ_NUM+1))
    el = np.rad2deg(np.arctan2(z,x))
    az = np.rad2deg(np.arctan2(y,x))
    rng = np.sqrt(x**2 + y**2 + z**2)
    idxEl = np.round((elMax - el)/elRes).astype('uint32')
    idxAz = np.round((azMax - az)/azRes).astype('uint32')        
    for iii in range(0,len(rng)):
        try:
            # img[idxEl[iii],idxAz[iii]] = 255-((rng[iii]/maxDist)*255).astype('uint8')
            if showLabel:
                if label[iii]==1:
                    img[idxEl[iii],idxAz[iii]] =255
                else:
                    img[idxEl[iii],idxAz[iii]] = 155
            else:
                img[idxEl[iii],idxAz[iii]] = intens[iii]
                # imgL[idxEl[iii],idxAz[iii]] = label[iii]
        except:
            print('out of range')
    
    #f = plt.figure(1)
    # plt.imshow(img)
    # plt.imshow(img-prev_img)
    kernel = np.ones((3,3), np.uint8)

        #img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    if diffFalg:
        plt.title('Frame #{}'.format(str(idx)))
        plt.imshow(img_dilation-prev_img)
        # if showLabel:
        #     plt.hold
        #     plt.imshow(imgL)
        #     plt.hold(False)

    else:
        plt.title('Frame #{}'.format(str(idx)))
        plt.imshow(img_dilation)
        # if showLabel:
        #     plt.hold(True)
        #     plt.imshow(imgL)
        #     plt.hold(False)
        if saveFig:
            plt.savefig('fig_'+str(idx)+'.png')
        
        # plt.t
    plt.pause(0.05)
    return img_dilation
    
def pc2imRef(x,y,z,ref,prev_img):
    img = np.zeros((EL_NUM,AZ_NUM))
    el = np.rad2deg(np.arctan2(z,x))
    az = np.rad2deg(np.arctan2(y,x))
    rng = np.sqrt(x**2 + y**2 + z**2)
    idxEl = np.round((elMax - el)/elRes).astype('uint32')
    idxAz = np.round((azMax - az)/azRes).astype('uint32')        
    for iii in range(0,len(rng)):
        try:
            # img[idxEl[iii],idxAz[iii]] = 255-((rng[iii]/maxDist)*255).astype('uint8')
            img[idxEl[iii],idxAz[iii]] = ref
            
        except:
            print('out of range')
    
    #f = plt.figure(1)
    # plt.imshow(img)
    kernel = np.ones((2,2), np.uint8)

        #img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    if diffFalg:
        plt.imshow(img-img_prev)
    else:
       
        plt.imshow(img_dilation)
    # plt.imshow(img-prev_img)
    plt.pause(0.05)
    return img


def pcTopView(x,y,z,d,label,img_prev,idx,diffFalg=False):
    
    img = np.zeros((EL_NUM,AZ_NUM))
    # el = np.rad2deg(np.arctan2(z,x))
    # az = np.rad2deg(np.arctan2(y,x))
    # rng = np.sqrt(x**2 + y**2 + z**2)
    idxY = np.round((maxDist_y/2+y)*pix2m_y).astype('uint32')
    idxX = np.round(x*pix2m_x).astype('uint32')   
    
    
    for iii in range(0,len(d)):
        try:
            if z[iii]>0.9 and z[iii]<1.8:
                if showLabel:
                    if label[iii]==1:
                        img[idxY[iii],idxX[iii]] =1
                    else:
                        img[idxY[iii],idxX[iii]] = 0
                else:
                    # img[idxY[iii],idxX[iii]] = max(img[idxY[iii],idxX[iii]],255-((d[iii])*255).astype('uint8'))
                    img[idxY[iii],idxX[iii]] = 1
                # img[idxY[iii],idxX[iii]] = 255
                
            # img[idxY[iii],idxX[iii]] = 255
        except:
            print('out of bounce')
    
    # tmp=np.where(img == 0,[img])

    
    #f = plt.figure(1)
    kernel = np.ones((2,2), np.uint8)
    kernel_erode = np.ones((2,2), np.uint8)

        #img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation2 = cv2.dilate(img, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation2, kernel_erode, iterations=1)
    img_dilation = cv2.dilate(img_erode, kernel_erode, iterations=1)
    if diffFalg:
        plt.imshow(img_dilation-img_prev)
    else:
        print('plotting: '+ str(idx))
        
        # plt.imshow(img_dilation)
    
    
    
    tmp=np.nonzero(img_dilation)
    X=np.concatenate(([tmp[0]],[tmp[1]]),axis=0)
    X=X.T
    global centroid,Cluster_std_vec
    centroid,Cluster_std_vec=clustMake(centroid,idx,X)
    plt.pause(0.01)
    return img_dilation,idxX,idxY

def evaluate_frame(gt_labels, pred_labels):

    assert np.all(np.isin(pred_labels, (0, 1))), \
        'Invalid values: pred labels value should be either 0 or 1, got {}'.format(set(pred_labels))
    print(np.sum(pred_labels))
    tp = 0
    fn = 0
    fp = 0
    for iii in range(0,len(gt_labels)):
       if  np.abs(gt_labels[iii] - int(pred_labels[iii])) > 0:
           if gt_labels[iii] == 1:
               fp = fp + 1
           else:
               fn = fn + 1
       else:
           if gt_labels[iii] == 1:
               tp = tp + 1
                   
    '''
    correct_predictions = gt_labels == pred_labels
    positive_predictions = pred_labels == 1
    # correct, positive prediction -> True positive
    tp = np.sum(correct_predictions & positive_predictions)

    # incorrect, negative prediction (using De Morgan's law) -> False negative
    fn = np.sum(np.logical_not(correct_predictions | positive_predictions))

    # incorrect, positive prediction -> False positive
    fp = np.sum(np.logical_not(correct_predictions) & positive_predictions)
    '''
    iou = tp/(tp+fn+fp)
    print(tp,fn,fp,iou)
    return iou
if __name__ == '__main__':
    global centroid,Cluster_std_vec
    Cluster_std_vec_arr = []
    centroid=[]
    centriodArr=[]
    base_dir = os.getcwd()
    #video_dir = os.path.join(base_dir, 'data_examples', 'test_video')
    # video_dir = os.path.join(base_dir, 'data', 'train', 'vid_1')
    # video_dir='D:\\DataHack18\\Dataset\\DataHack2018-master\\data_examples\\test_video'
    video_dir='D:\\DataHack18\\Dataset\\Train\\vid_1'
    video_dir='D:\\DataHack18\\Dataset\\Test\\Test\\vid_19'
    #import pdb;pdb.set_trace()
    frame_num = data_utils.count_frames(video_dir)
    min_idx = 0
    decimate = 1
    #elMin = 9999
    #azMin = 9999
    #elMax = -9999
    #azMax = -9999
    # agg_point_cloud_list=[]
    prev_img=np.zeros((EL_NUM,AZ_NUM)) # init
    imgSum = np.zeros((EL_NUM,AZ_NUM))
    prev_img_list=[]
    idxXY_list = []
    label_list = []
    fn_list = []
    for idx, frame in enumerate(data_utils.enumerate_frames(video_dir)):
        if idx < min_idx or idx % decimate != 0:
            continue
        pc, ego, label,label_pred_new,fn = data_utils.read_all_data(video_dir, frame)
        label=label_pred_new
        fn_list.append(fn)
        ego_rt = RotationTranslationData(vecs=(ego[:3], ego[3:]))
        ego_pc = ego_rt.apply_transform(pc[:, :3])
        ego_pc = np.concatenate((ego_pc, pc[:, 3:4]), -1)

        labeled_pc = np.concatenate((ego_pc, label), -1)
        # agg_point_cloud_list.append(labeled_pc)
        
        
        # img = np.zeros((EL_NUM,AZ_NUM))
        # # img1 = np.zeros((EL_NUM,AZ_NUM))
        # el = np.rad2deg(np.arctan2(pc[:,2],pc[:,0]))
        # az = np.rad2deg(np.arctan2(pc[:,1],pc[:,0]))
        # rng = np.sqrt(pc[:,0]**2 + pc[:,1]**2 + pc[:,2]**2)
        # idxEl = np.round((elMax - el)/elRes).astype('uint32')
        # idxAz = np.round((azMax - az)/azRes).astype('uint32')        
        # for iii in range(0,len(rng)):
        #     img[idxEl[iii],idxAz[iii]] = 255-((rng[iii]/100)*255).astype('uint8')
        # 
        # #f = plt.figure(1)
        # plt.imshow(img)
        # plt.pause(0.05)
        # x=ego_pc[:,0]
  
        if idx%KeyFrameUpdate==0:
            
            off_x=ego[3]
            off_y=ego[4]
            off_z=ego[5]
        #     prev_imgN=prev_img
        #     imgSum2Show = imgSum/imgSum.max()
        #     imgSum2Show[np.where(imgSum2Show > 0.1)] = 0
        #     plt.imshow(imgSum2Show)
        #     #plt.show()
        #     time.sleep(0.5)
        #     
        #     for idxTmp,prev_img_tmp in enumerate(prev_img_list):
        #         label_pred = np.zeros(len(label_list[idxTmp])) 
        #         idxXY_tmp = idxXY_list[idxTmp] 
        #         for iii in range(0,len(label_list[idxTmp])):
        #             if imgSum2Show[idxXY_tmp[1][iii],idxXY_tmp[0][iii]] > 0:
        #                 if prev_img_tmp[idxXY_tmp[1][iii],idxXY_tmp[0][iii]] > 0.7:
        #                     label_pred[iii] = 1
        #         iou = evaluate_frame(label_list[idxTmp],label_pred)
        #         if writeLabelFlag:
        #             np.savetxt(video_dir+"\\"+fn_list[idxTmp]+"_label_pred.csv", label_pred,fmt='%i', delimiter=",")
        #     imgSum = np.zeros((EL_NUM,AZ_NUM))
        #     prev_img_list=[]
        #     idxXY_list = []
        #     label_list = []
      
      
        if EGO_Flag:
            x=-(off_x-ego_pc[:,0])
            y=-(off_y-ego_pc[:,1])
            z=ego_pc[:,2]-off_z
            ref=ego_pc[:,3]
            label=labeled_pc[:,4]
        else:
            x=pc[:,0]
            y=pc[:,1]
            z=pc[:,2]
            ref=pc[:,3]
            label=pc[:,4]
        
        

        
        if idx%5==0:
            prev_imgN=prev_img
            
            
        # prev_img=pc2im_PerProj(norm_y,norm_z,ref,prev_img,idx,diffFalg=False)
        prev_img=pc2im(x,y,z,ref,label,prev_imgN,idx,diffFalg=False)
        # prev_img,idxX,idxY=pcTopView(x,y,z,ref,label,prev_imgN,idx,diffFalg=False)
        # imgSum+=prev_img
        # prev_img_list.append(prev_img)
        # label_list.append(label)
        # idxXY_list.append((idxX,idxY))
        # centriodArr.append(centroid)
        # Cluster_std_vec_arr.append(Cluster_std_vec)
        if idx > 250:
            break
## analyze centroids
# if False:
#     xCF=[]
#     yCF=[]
#     
#     # plt.figure(111)
#     
#     for kk in range(len(centriodArr)): # frame
#         
#         if kk >0:
#             clust_arr=centriodArr[kk]
#             clust_arr_p=centriodArr[kk-1]
#             for hh in range(len(clust_arr)):
#                 dxCL=clust_arr[hh][0]-clust_arr_p[hh][0]
#                 dyCL=clust_arr[hh][1]-clust_arr_p[hh][1]
#                 print('dx = '+str(dxCL))
#                 print('dy = '+str(dyCL))
#                 
#                 print('x ='+str(clust_arr[hh][0]))
#                 print('xp ='+str(clust_arr_p[hh][0]))
#                 print('y ='+str(clust_arr[hh][1]))
#                 print('yp ='+str(clust_arr_p[hh][1]))
#                 print('-----------------------------')
#     
#     
#     
#     
#     
#     for cF in centriodArr: # per frame
#         # print(cF[0])
#         for clus in cF: # per pixel
#             # xCF.append(clus[0])
#             # yCF.append(clus[1])
#             dX=cF
#             
#             
#             plt.plot(clus[0],clus[1],'o')
#             plt.xlim(0,EL_NUM)
#             plt.ylim(0,AZ_NUM)
#             plt.show()
#             plt.hold(True)
#             
#     for kk in range(len(centriodArr)):
#         print(centriodArr[kk])
#         print(Cluster_std_vec_arr[kk])