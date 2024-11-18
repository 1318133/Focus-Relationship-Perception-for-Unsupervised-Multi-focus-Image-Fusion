import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim, sigmoid, zeros_like
from tqdm import tqdm
from imageio import imwrite
from eval import eval_net

import torch.nn.functional as F

from unet import multifocusv3

import pytorch_ssim
from pytorch_msssim import ssim
import cv2

# from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, TestDataset
from torch.utils.data import DataLoader, random_split

# dir_img = '/home/s1u1/dataset/TNO/ir_128/'
# dir_mask = '/home/s1u1/dataset/TNO/vi_128/'
# dir_img = '/home/s1u1/dataset/roadscene/ir_128/'
# dir_mask = '/home/s1u1/dataset/roadscene/vi_128/'
# dir_img = '/home/s1u1/dataset/nirscene/ir_128/'
# dir_mask = '/home/s1u1/dataset/nirscene/vi_128/'
dir_img = '/home/s1u1/dataset/lytroDataset/A_enhance_128/'#'/home/s1u1/dataset/MFFW/A_gray_128/'#'/home/s1u1/dataset/RealMFF/imageA_gray128/'
dir_mask = '/home/s1u1/dataset/lytroDataset/B_enhance_128/'#'/home/s1u1/dataset/MFFW/B_gray_128/'#'/home/s1u1/dataset/RealMFF/imageB_gray128/'
# dir_img = '/home/s1u1/dataset/explosure/ll_gray_4_128/'
# dir_mask = '/home/s1u1/dataset/explosure/hh_gray_4_128/'


dir_checkpoint = 'checkpoints/'
showpathimg = 'epoch_fuseimg_show/img/'
showpathvis = 'epoch_fuseimg_show/vis/'
showpathinf = 'epoch_fuseimg_show/inf/'
showpathadd = 'epoch_fuseimg_show/add/'
showpathxo = 'epoch_fuseimg_show/xo/'
showpathyo = 'epoch_fuseimg_show/yo/'
showpathfxo = 'epoch_fuseimg_show/fxo/'
showpathfyo = 'epoch_fuseimg_show/fyo/'
showpathcf = 'epoch_fuseimg_show/cf/'
showpathbase = 'epoch_fuseimg_show/base/'

SSIM_WEIGHTS = [1, 10, 100, 1000]

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
loss_mse = nn.MSELoss(reduction='mean').cuda()

ups2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
avp2 = nn.AvgPool2d(8)

class grad1(nn.Module):

    def __init__(self):
        super(grad1, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        # enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  org_mean	

        # org_pool =  self.pool(org_mean)			
        # enhance_pool = self.pool(enhance_mean)
        # grad1 = features_grad(org_pool)
        # grad2 = features_grad(enhance_pool)
        # E = 1000*torch.pow(grad1-grad2,2)	

        # weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        # D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        # D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        # D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        # D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf,2)
        D_right = torch.pow(D_org_right,2)
        D_up = torch.pow(D_org_up,2)
        D_down = torch.pow(D_org_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

grad_set = grad1()

def average(features):
    kernel = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads

def features_grad1(features):
    kernel = [[0, 1, 0], [1, -1, 1], [0, 1, 0]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads

def features_grad2(f1,f2):

    b1 = torch.sigmoid(torch.abs(features_grad(f1)))
    b2 = torch.sigmoid(torch.abs(features_grad(f2)))
    
    c1 = b1 /(b1+b2)#+0.0000001)
    c2 = b2 /(b1+b2)#+0.0000001)
    return c1,c2

def dist_int(f1,f2):
    r = 100
    g1 = average(f1)
    g2 = average(f2)
    w1 = torch.abs(g1-f2)-torch.abs(f1-f2)
    
    w2 = (g2-f1)
    w1mean = w1.mean()
    w2mean = w2.mean()
    
    c1 = 1-torch.exp(-w1*r)
    c2 = 1-torch.exp(-w2*r)
    c1 = (c1-c1.min())/(c1.max()-c1.min())
    c2 = (c2-c2.min())/(c2.max()-c2.min())
    
    # # # w1 = (g1-f2)*(g1-f2)#(f1-f2)*(f1-f2)-
    # # # w2 = (g2-f1)*(g2-f1)#(f1-f2)*(f1-f2)-
    # mone = torch.ones_like(w1)
    mzero = torch.zeros_like(w1)
    # mhaf = torch.zeros_like(w1)*0.5
    # c1 = torch.where(w1>w1mean,mone,mzero)
    # # c1 = torch.where(w1==0,mhaf,c1)
    # c2 = torch.where(w2>w2mean,mone,mzero)
    # # c2 = torch.where(w2==0,mhaf,c2)
    b1 = torch.where(c1>c2,c1-c2,mzero)
    b2 = torch.where(c2>c1,c2-c1,mzero)
    # b1 = c1-c2
    # b2 = c2-c1
    
    return b1,b2,c1,c2#,g1,g2 #

def same_int(f1,f2):

    g1 = torch.abs(grad_set(f1))
    g2 = torch.abs(grad_set(f2))
    g11 = 1-(g1 - g1.min())/ (g1.max() - g1.min())
    g22 = 1-(g2 - g2.min())/ (g2.max() - g2.min())
    sm = torch.exp(-torch.abs(g11-g22)*10)
    g11 = torch.pow(g11,2)
    g22 = torch.pow(g22,2)
    out = sm*g11*g22
    return out

    # mone = torch.ones_like(g1)
    # mzero = torch.zeros_like(g1)
    # mhaf = torch.zeros_like(g1)*0.5
    # c1 = torch.where(g1>g2,mone,mzero)
    # c1 = torch.where(g1==g2,mhaf,c1)
    # c2 = 1-c1
    # b1 = torch.exp(-(1/(100*(f1-f2)*(f1-f2)+0.00000001)))
    # b1 = b1*c1
    # b2 = torch.exp(-(1/(100*(f1-f2)*(f1-f2)+0.00000001)))
    # b2 = b2*c2
    # # f_m = 0.5#torch.mean(f1+f2)
    # # # b1 = torch.exp(-torch.sigmoid(f1-f_m)*torch.sigmoid(f1-f_m)*12.5)
    # # # b2 = torch.exp(-torch.sigmoid(f2-f_m)*torch.sigmoid(f2-f_m)*12.5)
    # # b1 = torch.exp(-(f1-f_m)*(f1-f_m)*12.5)
    # # b2 = torch.exp(-(f2-f_m)*(f2-f_m)*12.5)
    # # g1 = g1 /(g1+g2+0.00000001)
    # # g2 = g2 /(g1+g2+0.00000001)
    # return b1,b2#,g1,g2

def features_grad3(f1,f2):
    # b1 = torch.abs(grad_set(f1))
    # b2 = torch.abs(grad_set(f2))
    b1 = torch.abs(features_grad(f1))
    b2 = torch.abs(features_grad(f2))
    # mone = torch.ones_like(b1)
    # mzero = torch.zeros_like(b1)
    # mhaf = torch.zeros_like(b1)*0.5
    # c1 = torch.where(b1>b2,mone,mzero)
    # c1 = torch.where(b1==b2,mhaf,c1)
    # c2 = 1-c1

    # c1 = b1 /(b1+b2+0.0000001)
    # c2 = b2 /(b1+b2+0.0000001)
    return b1,b2 #c1,c2


def features_int1(f1,f2):
    f_m = 0.5#torch.mean(f1+f2)
    # b1 = torch.exp(-torch.sigmoid(f1-f_m)*torch.sigmoid(f1-f_m)*12.5)
    # b2 = torch.exp(-torch.sigmoid(f2-f_m)*torch.sigmoid(f2-f_m)*12.5)
    b1 = torch.exp(-(f1-f_m)*(f1-f_m)*12.5)
    b2 = torch.exp(-(f2-f_m)*(f2-f_m)*12.5)
    c1 = b1 /(b1+b2)
    c2 = b2 /(b1+b2)
    return c1,c2

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)
        # grad1 = features_grad(org_pool)
        # grad2 = features_grad(enhance_pool)
        # E = 1000*torch.pow(grad1-grad2,2)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val): #(16, 0.6)
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
Lspa = L_spa()
Lexp = L_exp(16,0.6)
Ltv = L_TV()
def make_range_line( m1, m2):
    max1 = torch.max(m1)
    min1 = torch.min(m1)
    m_1 = (m1-min1)/(max1 -min1)
    max2 = torch.max(m2)
    min2 = torch.min(m2)
    m_2 = (m2-min2)/(max2 -min2)
    # m_1 = torch.tanh(m1)
    # m_2 = torch.tanh(m2)
    # m_1 = m_1 / (m_1 + m_2)
    # m_2 = m_2 / (m_1 + m_2)
    # m_1 = 1 - m_1
    return m_1, m_2

class grad(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, inchannel):
        super().__init__()
        kernel = torch.Tensor([[0, 1 / 4, 0], [1 / 4, -1, 1 / 4], [0, 1 /4, 0]])
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        self.conv = nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, padding_mode='replicate')
        
        self.conv.weight.data = kernel     

    def forward(self, x):
        return self.conv(x)

def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


def fusion_loss(img_1,img_2):
    # img1 = torch.from_numpy(np.rollaxis(img_1, 2)).float().unsqueeze(0)
    # img2 = torch.from_numpy(np.rollaxis(img_2, 2)).float().unsqueeze(0)

    if torch.cuda.is_available():
        img1 = img_1.cuda()
        img2 = img_2.cuda()
    # print(pytorch_ssim.ssim(img1, img2))
    ssimloss = pytorch_ssim.SSIM(window_size = 11)

    ssim_weight = SSIM_WEIGHTS[2]
    ssim_loss_value = ssimloss(img1, img2)
    # pixel_loss = mae_loss(img1,img2)
    pixel_loss = mse_loss(img1,img2)
    ssim_loss = 1 - ssim_loss_value
    # stay_loss = 5*ssim_loss + 10*pixel_loss
    stay_loss = ssim_loss + 20*pixel_loss

    return stay_loss

def imgshow(img, showpath, index):
    img = img[1,:,:,:]
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1,2,0)
    img = img.astype('uint8')
    if img.shape[2] == 1:
        img = img.reshape([img.shape[0], img.shape[1]])
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    return img

def imgshow1(img, showpath, index):
    # img = img.squeeze()
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1, 2, 3, 0)
    img = img.squeeze(0).astype('uint8')
    # img = img_final.transpose(1,2,0)
    # img = img.astype('uint8')
    if img.shape[2] == 1:
        img = img.reshape([img.shape[0], img.shape[1]])
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    return img

def caculateloss_v3(fxout, fyout, img, imgs, true_masks):
    m1 = features_grad(fxout)
    m1abs = torch.abs(m1)
    m2 = features_grad(fyout)
    m2abs = torch.abs(m2)
    mone = torch.ones_like(m1)
    mzero = torch.zeros_like(m1)
    mhaf = torch.zeros_like(m1)*0.5
    mask1 = torch.where(m1abs>m2abs,mone,mzero)
    mask1 = torch.where(m1abs==m2abs,mhaf,mask1)
    mask2 = 1-mask1
    weight1 = mask1#torch.sigmoid((m1abs-m2abs)*mask1*10000)+(1-torch.sigmoid((m2abs-m1abs)*mask2*10000))
    # weight1 = midfilt(weight1)
    # weight2 = mask2#torch.abs(torch.sigmoid((m1abs-m2abs)*mask1))+(1-torch.abs(torch.sigmoid((m2abs-m1abs)*mask2)))
    weight2 = mask2#1-weight1
    # torch.sigmoid(m1-torch.mean(m1))-torch.sigmoid(m2-torch.mean(m2))
    # weight1 = torch.sigmoid((torch.sqrt(torch.abs(m1))-torch.sqrt(torch.abs(m2)))*10)
    # bach, w, h = weight1.shape[0],weight1.shape[1],weight1.shape[2]
    # weight1 = torch.reshape(weight1, [bach,1,w,h])
    # weight2 = 1-weight1
    
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    # bb1 = torch.sigmoid(fxout)
    # bb2 = torch.sigmoid(fyout)
    x1 = torch.mean(fxout)
    x2 = torch.mean(fyout)
    # x1 = torch.mean(torch.mean(fxout,dim=2),dim=2)
    # x2 = torch.mean(torch.mean(fyout,dim=2),dim=2)
    # a = x1.shape[0]
    # x1 = torch.reshape(x1,[a,1,1,1])
    # x2 = torch.reshape(x2,[a,1,1,1])
    bb1 = torch.sqrt(torch.sigmoid(fxout-x1))
    bb2 = torch.sqrt(torch.sigmoid(fyout-x2))
    # bb1 = torch.abs(torch.mean(torch.tanh(fxout-x1)))
    # bb2 = torch.abs(torch.mean(torch.tanh(fyout-x2)))
    b1 = bb1/(bb1+bb2)
    b2 = bb2/(bb1+bb2)
    b1 = torch.sigmoid(b1-torch.mean(b1))
    b2 = torch.sigmoid(b2-torch.mean(b2))
    # b1 = 0.5
    # b2 = 0.5
    bw1 = img*b1
    bimgs = imgs*b1
    bw2 = img*b2
    btrue_masks = true_masks*b2
    loss_3 = loss_mse(bw1, bimgs) + loss_mse(bw2, btrue_masks)
    loss_3 = torch.mean(loss_3)

    loss = loss_1 + 20 * loss_2 + 0.4*loss_3
    return loss, weight1, weight2

def ssim_mse(img, imgs):

    loss_1 = (1 - ssim(img, imgs))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(img, imgs)
    loss_2 = torch.mean(loss_2)


    loss = loss_1 + 20 * loss_2 #+ 0.4*loss_3
    return loss#, imgs, true_masks

def base(fxout, fyout, img, imgs, true_masks):
    weight1 = fxout
    weight2 = fyout
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)


    loss = loss_1 + 20 * loss_2 #+ 0.4*loss_3
    return loss#, imgs, true_masks

def train_net(net, 
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.25):
    ph = 1
    al = 0 #低于此全部置0
    c = 3500

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train, val1 = random_split(dataset, [n_train, n_val])
    n_val9 = int(n_val*0.5)
    n_val1 = n_val-n_val9
    val2, val = random_split(val1, [n_val9, n_val1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    index = 1
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    # sloss = sobelloss()
    # sloss2 = sobelloss2()
    # usloss = unsameloss()
    up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    downmax2 = nn.MaxPool2d(2)
    downmax4 = nn.MaxPool2d(4)
    # down8 = nn.MaxPool2d(8)
    downavg2 = nn.AvgPool2d(2)
    downavg4 = nn.AvgPool2d(4)
    down8 = nn.AvgPool2d(8)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-7, momentum=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        net.train()

        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs2 = batch['image2']
                true_masks2 = batch['mask2']
                imgs3 = batch['image3']
                true_masks3 = batch['mask3']
                imgs4 = batch['image4']
                true_masks4 = batch['mask4']
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    imgs2 = imgs2.cuda()
                    true_masks2 = true_masks2.cuda()
                    imgs3 = imgs3.cuda()
                    true_masks3 = true_masks3.cuda()
                    imgs4 = imgs4.cuda()
                    true_masks4 = true_masks4.cuda()
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                imgs2 = imgs2.to(device=device, dtype=torch.float32)
                true_masks2 = true_masks2.to(device=device, dtype=torch.float32)
                imgs3 = imgs3.to(device=device, dtype=torch.float32)
                true_masks3 = true_masks3.to(device=device, dtype=torch.float32)
                imgs4 = imgs4.to(device=device, dtype=torch.float32)
                true_masks4 = true_masks4.to(device=device, dtype=torch.float32)

                outx,outy,wxh,wyh, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4 = net(imgs, true_masks)
                # out1,cx1,cy1= net(imgs, true_masks)

############像素级loss
                # baseout = true_masks*wxh+imgs*wyh
                
                
                # l_same1 = ssim_mse(outx*wyh,true_masks*wyh)#out)
                # l_same2 = ssim_mse(outy*wxh,imgs*wxh)
                l_g = torch.exp(-torch.mean(torch.abs(grad_set(true_masks*wyh+imgs*wxh))-torch.abs(grad_set(imgs*wyh+true_masks*wxh)))*100)
                # l_g2 = torch.exp(-torch.mean(torch.abs(grad_set(true_masks*cx4+imgs*cx3))-torch.abs(grad_set(imgs*cx4+true_masks*cx3)))*100)
                # l_g3 = torch.exp(-torch.mean(torch.abs(grad_set(true_masks*cy4+imgs*cy3))-torch.abs(grad_set(imgs*cy4+true_masks*cy3)))*100)
                # xx = same_int(imgs, true_masks)
                g_i = torch.abs(grad_set(imgs))
                g_ii = torch.exp(-1000*((g_i-g_i.min())/(g_i.max()-g_i.min())))
                g_t = torch.abs(grad_set(true_masks))
                g_tt = torch.exp(-1000*((g_t-g_t.min())/(g_t.max()-g_t.min())))

                w_g = torch.abs(grad_set(wxh))
                w_ga = average(average(average((w_g-w_g.min())/(w_g.max()-w_g.min()))))
                w_gas = average(w_ga*(torch.exp(-10000*torch.abs(g_i-g_t))))

                w_gas1 = torch.tanh(10*((w_gas-w_gas.min())/(w_gas.max()-w_gas.min())))
                maxx = torch.tanh((up(downavg4(wxh))-0.5)*10)
                maxx = (maxx-maxx.min())/(maxx.max()-maxx.min())
                # w_fi = wxh*(1-w_gas1)+w_gas1*(maxx)#
                w_fi2 = maxx*(1-w_gas1)+w_gas1*(0.5)#wxh+w_gas1
                onesw_fi = torch.ones_like(w_fi2)
                w_fi2 = torch.where(w_fi2 > onesw_fi, onesw_fi, w_fi2)
                w_fi3 = 1-w_fi2
                # g_tt = 1-((g_t-g_t.min())/(g_t.max()-g_t.min()))
                # g_h = torch.abs(grad_set(g_t*wyh+g_i*wxh))
                # g_l = torch.abs(grad_set(g_i*wyh+g_t*wxh))

                # l_diviation = torch.sigmoid(-(torch.mean(torch.pow(g_h-torch.mean(g_h),2))-torch.mean(torch.pow(g_l-torch.mean(g_l),2))))
                # l_diviation = torch.sigmoid(-100*(torch.mean(torch.pow(g_h-torch.mean(g_h),2))-torch.mean(torch.pow(g_l-torch.mean(g_l),2))))
                out = true_masks*w_fi3+imgs*w_fi2 #true_masks*wyh+imgs*wxh
                onesout = torch.ones_like(out)
                zerosout = torch.zeros_like(out)
                i_out = torch.where(g_i > g_t, onesout, zerosout)
                t_out = 1-i_out
                
                i_or = torch.exp(-torch.abs(imgs-true_masks))
                i_cang = torch.exp(-torch.abs(average(imgs)-true_masks))
                i_oc = torch.where(i_cang > i_or, onesout, zerosout)
                t_or = torch.exp(-torch.abs(true_masks-imgs))
                t_cang = torch.exp(-torch.abs(average(true_masks)-imgs))
                t_oc = torch.where(t_cang > t_or, onesout, zerosout)
                i_set = torch.mean(1-wxh*g_ii*i_oc)
                # i_mset = 1-wxh*g_ii*i_oc#1-i_out*g_ii*i_oc
                t_set = torch.mean(1-wyh*g_tt*t_oc)
                # t_mset = 1-wyh*g_tt*t_oc#1-(1-i_out)*g_tt*t_oc

                # out = true_masks*wyh+imgs*wxh
                l_cf = ssim_mse(cy2,out)

                # g_out = 1-torch.mean(g_out)
                # Loss_TV = Ltv(cy2)
                # loss_exp = torch.mean(Lexp(cy2))
                # l_same3 = ssim_mse(outx,outy)
                # # l_same2 = mae_loss(out1,true_masks)

                # xwg1,ywg1 = features_grad3(imgs,true_masks)#imgs, true_masks)
                # # xwg2,ywg2 = features_grad3(cx2,cy2)#imgs2, true_masks2)
                # # xwg3,ywg3 = features_grad3(cx3,cy3)#imgs3, true_masks3)
                # # xwg4,ywg4 = features_grad3(cx4,cy4)#imgs4, true_masks4)

                # xwi1,ywi1 = features_int1(cx1,cy1)#imgs, true_masks)
                # # xwi2,ywi2 = features_int1(cx2,cy2)#imgs2, true_masks2)
                # # xwi3,ywi3 = features_int1(cx3,cy3)#imgs3, true_masks3)
                # # xwi4,ywi4 = features_int1(cx4,cy4)#imgs4, true_masks4)

                # xwg11,ywg11 = features_grad3(imgs, true_masks)
                # # xwg22,ywg22 = features_grad3(imgs2, true_masks2)
                # # xwg33,ywg33 = features_grad3(imgs3, true_masks3)
                # # xwg44,ywg44 = features_grad3(imgs4, true_masks4)

                # xwi11,ywi11 = same_int(imgs, true_masks)#,gr1,gr2
                # asd1,asd2,asd3,asd4 = dist_int(imgs, true_masks)
                # xwi22,ywi22 = features_int1(imgs2, true_masks2)
                # xwi33,ywi33 = features_int1(imgs3, true_masks3)
                # xwi44,ywi44 = features_int1(imgs4, true_masks4)
                # xwg22 = down2(xwg11)
                # ywg22 = down2(ywg11)
                # xwg33 = down4(xwg11)
                # ywg33 = down4(ywg11)
                # xwg44 = down8(xwg11)
                # ywg44 = down8(ywg11)
                # # l_same1 = mae_loss(xwg1,xwg11)+mae_loss(ywg1,ywg11)
                # # l_same2 = mae_loss(xwg2,xwg22)+mae_loss(ywg2,ywg22)
                # # l_same3 = mae_loss(xwg3,xwg33)+mae_loss(ywg3,ywg33)
                # # l_same4 = mae_loss(xwg4,xwg44)+mae_loss(ywg4,ywg44)

                # alph = 0
                # bata = 1

                # xw1 = bata*xwg11 + alph*xwi1
                # # xw2 = bata*xwg22 + alph*xwi2
                # # xw3 = bata*xwg33 + alph*xwi3
                # # xw4 = bata*xwg44 + alph*xwi4

                # yw1 = bata*ywg11 + alph*ywi1
                # # yw2 = bata*ywg22 + alph*ywi2
                # # yw3 = bata*ywg33 + alph*ywi3
                # # yw4 = bata*ywg44 + alph*ywi4

                # l1 = base(xw1, yw1, out1, imgs, true_masks)#+alph*base(xwi1, ywi1, out1, imgs, true_masks)
                # l_same1 = mae_loss(asd2*out1,asd2*imgs)+mae_loss(asd1*out1,asd1*true_masks)
                # # l_same2 = mae_loss(xwi11*out1,xwi11*imgs)+mae_loss(ywi11*out1,ywi11*true_masks)
                # # l2 = base(xw2, yw2, out2, imgs2, true_masks2)#+alph*base(xwi2, ywi2, out2, imgs2, true_masks2)
                # # l3 = base(xw3, yw3, out3, imgs3, true_masks3)#+alph*base(xwi3, ywi3, out3, imgs3, true_masks3)
                # # l4 = base(xw4, yw4, out4, imgs4, true_masks4)#+alph*base(xwi4, ywi4, out4, imgs4, true_masks4)

                # # wx = torch.mean(fxout)
                # # wy = torch.mean(fyout)
                # # wx,wy = make_range_line(wx, wy)

                # # loss_spa1 = torch.mean(Lspa(out1, imgs+true_masks))
                # # loss_exp1 = 10*torch.mean(Lexp(out1))
                # # loss_spa2 = torch.mean(Lspa(inf, imgs+true_masks))
                # # loss_exp2 = 10*torch.mean(Lexp(inf))
                # # Loss_TV1 = 200*Ltv(xout)
                # # Loss_TV2 = 200*Ltv(yout)
                # # l_same = mae_loss(vis,inf)
                # # l_same1 = base(fxout, fyout, img, imgs, true_masks)#mae_loss(img,imgs)
                # # l_same2 = mae_loss(img,true_masks)
                # # fxout.requires_grad_(True)
                # # fyout.requires_grad_(True)
                # # l_same = mae_loss(fxout*vis,fyout*inf)
                # # l_same1 = mae_loss(vis,imgs)
                # # l_same2 = mae_loss(inf,true_masks)
                # # l_out = base(xout, yout, img, imgs,true_masks)

                # # loss1, weight1, weight2 = caculateloss(fxout, fyout, img, imgs, true_masks)
                # # loss1, weight1, weight2 = caculateloss_v3(fxout, fyout, img, imgs, true_masks)
                # # loss1, weight1, weight2 = base(fxout, fyout, img, imgs, true_masks)

                # # lossall = loss_spa1 + loss_spa2 + loss_exp1 + loss_exp2 + Loss_TV1 + Loss_TV2 #+ l_same# + 0.01*l_same1#*wx + l_same2*wy
                alph = 0.1
                lossall = l_g+(i_set+t_set)*alph#+l_cf##+Loss_TV l_same1+l_same2++l_g2+l_g3+10*loss_exp # +l_diviation#+l_same3#+Loss_TV## + 10*l_same1#+l2+l3+l4#+l_same2 + l_same1 + l_same3+l_same4# +loss_spa1 + loss_exp1 ##  

                pbar.set_postfix(**{'loss1': l_g.item(),'loss2': l_g.item(),'loss3': l_cf.item(),'loss4': i_set.item(),'loss5': t_set.item()})#,'loss6': l_same2.item(),'loss3 (batch)': loss3.item()})
                # pbar.set_postfix(**{'loss1': l1.item(),'loss2': l_same1.item(),'loss3': l_same2.item(),'loss4': l_same3.item(),'loss5': l_same4.item()})#,'loss6': l_same2.item(),'loss3 (batch)': loss3.item()})
                optimizer.zero_grad()
                # loss2.backward(retain_graph=True)
                # loss1.backward()
                # loss3.backward()
                lossall.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                ######
                if global_step == ((n_train // batch_size)*index):
                    onesout = torch.ones_like(out)
                    out = torch.where(out > onesout, onesout, out)
                    g = imgshow(out,showpathimg,index)
                    g1 = imgshow(w_gas,showpathvis,index)
                    g2 = imgshow(maxx,showpathinf,index)
                    g = imgshow(wxh,showpathxo,index)
                    g = imgshow(wyh,showpathyo,index)
                    g3 = imgshow(imgs,showpathfxo,index)
                    g4 = imgshow(true_masks,showpathfyo,index)

                    g = imgshow(cy2,showpathcf,index)
                    g = imgshow(cy1,showpathbase,index)
                    imgadd = g3*0.5 +g4*0.5
                    indexd = format(index, '05d')
                    file_name = str(indexd) + '.png'
                    path_out = showpathadd + file_name
                    imwrite(path_out, imgadd)
                    print(optimizer.state_dict()['param_groups'][0]['lr'])
#################
                    index += 1  
                #####
                if global_step % (n_train // (1  * batch_size)) == 0:
                # if global_step == 5:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        # value.requires_grad = False
                        # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        # writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        # writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                    #     writer.add_images('images/pred', back_pred, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=str, default=True,
                        help='If test images turn True, train images turn False')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    pthf = '/home/s1u1/code/multifocus_byweightsdesign/checkpointsall/newnet_lg_iset_tset_usewxh_beind4conv/CP_epoch50_v2.pth'

    # ir = '/home/s1u1/dataset/dajiang_yu2/1ch_inf/'
    # vi = '/home/s1u1/dataset/dajiang_yu2/1ch_rgb/'

    # ir = '/home/s1u1/dataset/yiyuan/vi/'
    # vi = '/home/s1u1/dataset/yiyuan/ir/'

    # ir = '/home/s1u1/dataset/TNO/ir/'
    # vi = '/home/s1u1/dataset/TNO/vi/'
    # ir = '/home/s1u1/dataset/explosure/hh_gray_4/'
    # vi = '/home/s1u1/dataset/explosure/ll_gray_4/'
    # ir = '/home/s1u1/dataset/nirscene/ir/'
    # vi = '/home/s1u1/dataset/nirscene/vi/'
    # ir = '/home/s1u1/dataset/roadscene/ir_1ch/'
    # vi = '/home/s1u1/dataset/roadscene/vi_1ch/'
    ir = '/home/s1u1/dataset/lytroDataset/A/'
    vi = '/home/s1u1/dataset/lytroDataset/B/'
    # ir = '/home/s1u1/dataset/RealMFF/imageA_testgray/'
    # vi = '/home/s1u1/dataset/RealMFF/imageB_testgray/'
    # ir = '/home/s1u1/dataset/MFI-WHU-master/MFI-WHU/B_gray20/'
    # vi = '/home/s1u1/dataset/MFI-WHU-master/MFI-WHU/A_gray20/'
    # ir = '/home/s1u1/dataset/MFFW/input/a_gray/'#'/home/s1u1/dataset/MFFW/A_gray/'
    # vi = '/home/s1u1/dataset/MFFW/input/b_gray/'#'/home/s1u1/dataset/MFFW/B_gray/'
    # path = './outputsall/nirsence/'
    path = '/home/s1u1/code/multifocus_byweightsdesign/ablation/loss_alph/01/chongxinpao/results/'#'./outputs/'#'/home/s1u1/code/multifocus_byweightsdesign/outputs_newnet_lg_lexp/'
    pathadd = './outputsadd/'
    path1 = './ir_outputs/'
    path2 = './vi_outputs/'

    dataset = TestDataset(ir, vi)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = multifocusv3.wtNet(n_channels=1, n_classes=1, bilinear=True, pthfile=pthf)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    index = 1
    Time = 0
    up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    downmax2 = nn.MaxPool2d(2)
    downmax4 = nn.MaxPool2d(4)
    # down8 = nn.MaxPool2d(8)
    downavg2 = nn.AvgPool2d(2)
    downavg4 = nn.AvgPool2d(4)
    wxhpath = '/home/s1u1/code/multifocus_byweightsdesign/outwxh/'
    wyhpath = '/home/s1u1/code/multifocus_byweightsdesign/outwyh/'
    outxpath = '/home/s1u1/code/multifocus_byweightsdesign/outx/'
    outypath = '/home/s1u1/code/multifocus_byweightsdesign/outy/'
    outisetpath = '/home/s1u1/code/multifocus_byweightsdesign/outiset/'
    outtsetpath = '/home/s1u1/code/multifocus_byweightsdesign/outtset/'
    conx1 = '/home/s1u1/code/multifocus_byweightsdesign/conv_x/cx1/'
    conx2 = '/home/s1u1/code/multifocus_byweightsdesign/conv_x/cx2/'
    conx3 = '/home/s1u1/code/multifocus_byweightsdesign/conv_x/cx3/'
    conx4 = '/home/s1u1/code/multifocus_byweightsdesign/conv_x/cx4/'
    if args.test:
        for im in test_loader:
            
            ir = im['image']
            vi = im['mask']
            if torch.cuda.is_available():
                ir = ir.cuda()
                vi = vi.cuda()
            # Net = Wdenet.wtNet(1, 1)
            # Net = unet_model.UNet(1,1)
            Net = multifocusv3.wtNet(1, 1, pthfile=pthf)
            Net = Net.cuda()
            ##########################
            add = ir*0.5 + vi*0.5
            img_final = add.detach().cpu().numpy()
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(1, 2, 3, 0)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = pathadd + file_name
            # index += 1            
            imwrite(path_out, img)
            ##########################
            start = time.time()
            outx,outy,wxh,wyh, cx1,cx2,cx3,cx4, cy1,cy2,cy3,cy4 = Net(vi, ir)
            g_i = torch.abs(grad_set(vi))
            g_ii = torch.exp(-1000*((g_i-g_i.min())/(g_i.max()-g_i.min())))
            g_t = torch.abs(grad_set(ir))
            g_tt = torch.exp(-1000*((g_t-g_t.min())/(g_t.max()-g_t.min())))

            w_g = torch.abs(grad_set(wxh))
            w_ga = average(average(average((w_g-w_g.min())/(w_g.max()-w_g.min()))))
            w_gas = average(w_ga*(torch.exp(-10000*torch.abs(g_i-g_t))))

            w_gas1 = torch.tanh(10*((w_gas-w_gas.min())/(w_gas.max()-w_gas.min())))
            maxx = torch.tanh((up(downavg4(wxh))-0.5)*10)
            maxx = (maxx-maxx.min())/(maxx.max()-maxx.min()+0.0000001)
            w_fi = wxh*(1-w_gas1)+w_gas1*(maxx)#
            w_fi2 = maxx*(1-w_gas1)+w_gas1*(0.5)#wxh+w_gas1
            onesw_fi = torch.ones_like(w_fi2)
            w_fi2 = torch.where(w_fi2 > onesw_fi, onesw_fi, w_fi2)
            w_fi3 = 1-w_fi2




            # w_g = torch.abs(grad_set(wxh))
            # w_ga = average(average(average((w_g-w_g.min())/(w_g.max()-w_g.min()))))
            # w_gas = average(w_ga*(torch.exp(-10000*torch.abs(g_i-g_t))))

            # w_gas1 = torch.tanh(10*((w_gas-w_gas.min())/(w_gas.max()-w_gas.min())))
            # maxx = torch.tanh((up(downavg4(wxh))-0.5)*10)
            # maxx = (maxx-maxx.min())/(maxx.max()-maxx.min())
            #     # w_fi = wxh*(1-w_gas1)+w_gas1*(maxx)#
            # w_fi2 = maxx*(1-w_gas1)+w_gas1*(0.5)#wxh+w_gas1
            # onesw_fi = torch.ones_like(w_fi2)
            # w_fi2 = torch.where(w_fi2 > onesw_fi, onesw_fi, w_fi2)
            # w_fi3 = 1-w_fi2
            #     # g_tt = 1-((g_t-g_t.min())/(g_t.max()-g_t.min()))
            #     # g_h = torch.abs(grad_set(g_t*wyh+g_i*wxh))
            #     # g_l = torch.abs(grad_set(g_i*wyh+g_t*wxh))

            #     # l_diviation = torch.sigmoid(-(torch.mean(torch.pow(g_h-torch.mean(g_h),2))-torch.mean(torch.pow(g_l-torch.mean(g_l),2))))
            #     # l_diviation = torch.sigmoid(-100*(torch.mean(torch.pow(g_h-torch.mean(g_h),2))-torch.mean(torch.pow(g_l-torch.mean(g_l),2))))
            # out = ir*w_fi3+vi*w_fi2 #true_masks*wyh+imgs*wxh
            onesout = torch.ones_like(ir)
            zerosout = torch.zeros_like(ir)
            i_out = torch.where(g_i > g_t, onesout, zerosout)
            t_out = 1-i_out
                
            i_or = torch.exp(-torch.abs(ir-vi))
            i_cang = torch.exp(-torch.abs(average(ir)-vi))
            i_oc = torch.where(i_cang > i_or, onesout, zerosout)
            t_or = torch.exp(-torch.abs(vi-ir))
            t_cang = torch.exp(-torch.abs(average(vi)-ir))
            t_oc = torch.where(t_cang > t_or, onesout, zerosout)
            # i_set = torch.mean(1-wxh*g_ii*i_oc)
            i_mset = i_out*g_ii*i_oc#1-wxh*g_ii*i_oc#
            # t_set = torch.mean(1-wyh*g_tt*t_oc)
            t_mset = 1-(1-i_out)*g_tt*t_oc#1-wyh*g_tt*t_oc#



            out = ir*w_fi3+vi*w_fi2 #true_masks*wyh+imgs*wxh
            onesout = torch.ones_like(out)
            # zerosout = torch.zeros_like(out)

            # out = ir*wyh+vi*wxh
            # onesout = torch.ones_like(out)
            out = torch.where(out > onesout, onesout, out)

            # wx,wy = make_range_line(r, j)
            img = out#+j #vi*(1-wx) + ir*(1-wy)
            # img,_,_,_,_,_,_ = Net(vi, ir)
            # img = Net(vi, ir)
            img_final = img.detach().cpu().numpy() 
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(1, 2, 3, 0)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            end = time.time()
            g = imgshow1(wxh,wxhpath,index)
            g1 = imgshow1(wyh,wyhpath,index)
            g = imgshow1(w_fi2,outxpath,index)
            g1 = imgshow1(w_fi3,outypath,index)
            g = imgshow1(i_mset,outisetpath,index)
            g1 = imgshow1(t_mset,outtsetpath,index)
            g = imgshow1(cx1,conx1,index)
            g1 = imgshow1(cx2,conx2,index)
            g = imgshow1(cx3,conx3,index)
            g1 = imgshow1(cx4,conx4,index)
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = path + file_name
            # index += 1            
            imwrite(path_out, img)
            Time += end-start
            print(index)
            print(end-start)
            index += 1  
        average_time = Time/(len(test_loader))  
        print(average_time) 

    else:
        try:
            train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
