import torch
from torch.utils import data
from PIL import Image
from scipy import ndimage
import numpy as np
import os
from main_code.data_lib.data_utils import *
import matplotlib.pyplot as plt
import json
import random
import cv2
from collections import OrderedDict
class YTBVOS_DATASET(data.Dataset):
    def __init__(self,mode):
        super(YTBVOS_DATASET, self).__init__()

        assert mode in ['train','valid']

        self.mode = mode
        self.data_path = '/mnt/sda1/don/documents/project_paper/video_seg/data/youtube_vos/'

        meta_path = self.data_path+'{}/meta.json'.format(self.mode)
        with open(meta_path,'r') as f:
            self._seqs = json.load(f)['videos']


        self.object_seqs = self._get_img_list()

        self.mean_value = [0,0,0]#[104,117,123]

        self.len = len(self.object_seqs)

        print(self.len)


    def __getitem__(self, index):

        object_name = self.object_seqs[index]
        object_frames = self.object_list[object_name]


        object_id = object_name.split('_')[0]
        object_lid = object_name.split('_')[1]

        if self.mode=='train':
            frames_num = len(object_frames)
            select_num = min(7,frames_num-1)
            select_frame = random.sample(object_frames[1:],select_num)
            select_frame = sorted(select_frame)

            init_image = np.array(Image.open(os.path.join(self.data_path,self.mode,'JPEGImages',object_id,object_frames[0]+'.jpg')))
            init_label = self._load_object_label(os.path.join(self.data_path,self.mode,'Annotations',object_id,object_frames[0]+'.png'),object_lid)
            init_image,init_label = self._resize_submean(init_image,(448,256),init_label)
            img_seqs = []
            ano_seqs = []
            for frame in select_frame:
                img_seq = np.array(Image.open(os.path.join(self.data_path,self.mode,'JPEGImages',object_id,frame+'.jpg')))
                ano_seq = self._load_object_label(os.path.join(self.data_path,self.mode,'Annotations',object_id,frame+'.png'),object_lid)

                img_seq, ano_seq = self._resize_submean(img_seq, (448, 256), ano_seq)

                img_seq = np.transpose(img_seq,(2,0,1))
                ano_seq = ano_seq[np.newaxis,...]
                img_seqs.append(img_seq)
                ano_seqs.append(ano_seq)

            img_seqs = np.stack(img_seqs,0)
            ano_seqs = np.stack(ano_seqs, 0)


            init_image_label = np.concatenate([init_image,init_label[...,np.newaxis]],-1)
            init_image_label = np.transpose(init_image_label,(2,0,1))
            init_image_label = torch.from_numpy(init_image_label).float()
            img_seqs = torch.from_numpy(img_seqs).float()
            ano_seqs = torch.from_numpy(ano_seqs).float()

            return init_image_label,img_seqs,ano_seqs

        else:

            init_image = np.array(
                Image.open(os.path.join(self.data_path, self.mode, 'JPEGImages', object_id, object_frames[0] + '.jpg')))
            init_label = self._load_object_label(
                os.path.join(self.data_path, self.mode, 'Annotations', object_id, object_frames[0] + '.png'),
                object_lid)
            init_image, init_label = self._resize_submean(init_image, (448, 256), init_label)
            img_seqs = []

            for frame in object_frames[1:]:
                img_seq = np.array(
                    Image.open(os.path.join(self.data_path, self.mode, 'JPEGImages', object_id, frame + '.jpg')))


                img_seq = self._resize_submean(img_seq, (448, 256))

                img_seq = np.transpose(img_seq, (2, 0, 1))

                img_seqs.append(img_seq)


            img_seqs = np.stack(img_seqs, 0)


            init_image_label = np.concatenate([init_image, init_label[..., np.newaxis]], -1)
            init_image_label = np.transpose(init_image_label, (2, 0, 1))
            init_image_label = torch.from_numpy(init_image_label).float()
            img_seqs = torch.from_numpy(img_seqs).float()


            return init_image_label, img_seqs


    def __len__(self):

        return self.len


    def _load_object_label(self,path,object_lid):

        anno = Image.open(path)
        anno = np.array(anno)
        anno = (anno==np.uint8(object_lid)).astype(np.uint8)

        return anno

    def _resize_submean(self,img,size,ano=None):

        img = cv2.resize(img,dsize=size,interpolation=cv2.INTER_CUBIC).astype(np.float32)

        img -= self.mean_value

        if ano is not None:
            ano = cv2.resize(ano,dsize=size,interpolation=cv2.INTER_NEAREST)

            return img,ano

        return img




    def _get_img_list(self):

        self.object_list = OrderedDict()
        for vid_id, seq in self._seqs.items():

            vid_objects = seq['objects']
            for label_id, obj_info in vid_objects.items():
                frames = obj_info['frames']
                if len(frames)<2:
                    continue
                self.object_list[vid_id+'_'+label_id] = frames

        object_seqs = list(self.object_list.keys())

        object_seqs = sorted(object_seqs)



        return object_seqs

    def _resize_padding(self,img,mask,size):

        height = img.shape[0]
        width = img.shape[1]
        scale = float(size)/max(height,width)
        imgr = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        maskr = cv2.resize(mask,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)

        height_n = imgr.shape[0]
        width_n = imgr.shape[1]

        all_pad = size-min(height_n,width_n)
        s0 = all_pad//2
        s1 = all_pad-s0

        if height_n<=width_n:
            imgr = cv2.copyMakeBorder(imgr,s0,s1,0,0,cv2.BORDER_CONSTANT,value=self.mean_value)
            maskr = cv2.copyMakeBorder(maskr,s0,s1,0,0,cv2.BORDER_CONSTANT,value=0)
        else:
            imgr = cv2.copyMakeBorder(imgr,0,0,s0,s1,cv2.BORDER_CONSTANT,value=self.mean_value)
            maskr = cv2.copyMakeBorder(maskr,0,0,s0,s1,cv2.BORDER_CONSTANT,value=0)


        return imgr,maskr


if __name__ == '__main__':


    dataset = YTBVOS_DATASET(mode='valid')

    data_iter = data.DataLoader(dataset,
                                batch_size=1,
                                drop_last=False,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True)
    data_iter = iter(data_iter)

    for init_image_label,img_seqs in data_iter:

        print(img_seqs.size(1))
        plt.figure(figsize=(30,30))
        plt.subplot(221)
        plt.imshow(init_image_label[0,0,:,:])
        plt.imshow(init_image_label[0,-1,:,:],alpha=0.5)

        plt.subplot(222)
        plt.imshow(img_seqs[0, 0, 0,:, :])
        #plt.imshow(ano_seqs[0, 0, 0,:, :],alpha=0.5)

        plt.subplot(223)
        plt.imshow(img_seqs[0, 1, 0,:, :])
        #plt.imshow(ano_seqs[0, 1, 0,:, :],alpha=0.5)

        plt.subplot(224)
        plt.imshow(img_seqs[0, 2, 0,:, :])
        #plt.imshow(ano_seqs[0, 2, 0,:, :],alpha=0.5)


        plt.show()


