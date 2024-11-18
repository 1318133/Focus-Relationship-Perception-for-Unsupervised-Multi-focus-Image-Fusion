import torch.nn.functional as F
import torch.nn as nn
# from unet.unet_parts import *
import numpy
import torch



up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
downmax2 = nn.MaxPool2d(2)
downmax4 = nn.MaxPool2d(4)
# down8 = nn.MaxPool2d(8)
downavg2 = nn.AvgPool2d(2)
downavg4 = nn.AvgPool2d(4)
down8 = nn.AvgPool2d(8)

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


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf,2)
        D_right = torch.pow(D_org_right,2)
        D_up = torch.pow(D_org_up,2)
        D_down = torch.pow(D_org_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
grad_set = grad1()


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

'''make for fusion'''
def fusion( en1, en2):
    f_0 = (en1 + en2)/2
    return f_0

def make_range_line( m1, m2):
    # max1 = torch.max(m1)
    # min1 = torch.min(m1)
    # m_1 = (m1-min1)/(max1 -min1)
    # max2 = torch.max(m2)
    # min2 = torch.min(m2)
    # m_2 = (m2-min2)/(max2 -min2)
    m_1 = torch.sigmoid(m1)
    m_2 = torch.sigmoid(m2)
    m_1 = m_1 / (m_1 + m_2)
    m_2 = m_2 / (m_1 + m_2)
    m_1 = 1 - m_1
    return m_1, m_2


def specific_size( m1, m2):

    one = torch.ones_like(m1) 
    # halfone = torch.ones_like(m1)*0.5
    zero = torch.zeros_like(m1)
    out = torch.where(m1 > m2, one, zero) 
    # out = torch.where(m1 == m2, halfone, out) 

    return out

def make_range( m1, m2):
    max1 = torch.mean(m1)
    max1 = max1 * 2

    max2 = torch.mean(m2)
    max2 = max2 * 2

    weigh1 = torch.ones_like(m1) * max1
    weigh2 = torch.ones_like(m2) * max2
    m_1 = torch.where(m1 > max1, weigh1, m1) 
    m_2 = torch.where(m2 > max2, weigh2, m2) 

    min1 = torch.min(m_1)
    m_1 = (m_1-min1)/(max1 -min1)
    min2 = torch.min(m_2)
    m_2 = (m_2-min2)/(max2 -min2)

    # m_1 = torch.sigmoid(m_1)
    # m_2 = torch.sigmoid(m_2)
    # m1 = 1 - m1   
    # m_1 = torch.sin(m1)
    # m_2 = torch.sin(m2)
    # m_1 = 1 - m_1    
    m_1 = m_1 / (m_1 + m_2)
    m_2 = m_2 / (m_1 + m_2)

    w1 = torch.ones_like(m_1) * 0.5
    w2 = torch.ones_like(m_2) * 0.5

    m_1 = torch.where(torch.isnan(m_1), w1, m_1) 
    m_2 = torch.where(torch.isnan(m_2), w2, m_2)     

    # m_1 = m1 / (m1 + m2)
    # m_2 = m2 / (m1 + m2)

    return m_1, m_2


def make_single_range(m1):
    max = torch.max(m1)
    min = torch.min(m1)
    m_1 = (m1-min)/(max -min)
    # m_1 = torch.sin(m1-0.5)+1
    m_1 = torch.sigmoid(m1)
    m_2 = 1 - m_1
    return m_2, m_1

def mask_fusion( x1, m1, x2, m2):
    mf = x1 * m1 +  x2 * m2
    return mf

def mask_addition( x1, x2):
    mf = x1 * 0.5 +  x2 * 0.5
    return mf

def Concat(x, y, z):
    return torch.cat((x, y), z)

class wtNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pthfile = 'F:/3line/checkpoints/CP_epoch13.pth'):
        super(wtNet, self).__init__()
         
        # net= UNet(n_channels=1, n_classes=1, bilinear=True)
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        number_f = 32
        h_num = 1

        self.c1_ia = nn.Conv2d(1,h_num,3,1,1,bias=True,padding_mode='reflect') 
        # self.c3_ia = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 
        # self.c5_ia = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 
        # self.c7_ia = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 
        self.c1_ib = nn.Conv2d(1,h_num,3,1,1,bias=True,padding_mode='reflect') 
        # self.c3_ib = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 
        # self.c5_ib = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 
        # self.c7_ib = nn.Conv2d(1,number_f/2,3,1,1,bias=True,padding_mode='reflect') 

        self.c1_a = nn.Conv2d(h_num,h_num,1,1,bias=True) 
        self.c3_a = nn.Conv2d(h_num,h_num,3,1,1,bias=True,padding_mode='reflect') 
        self.c5_a = nn.Conv2d(h_num,h_num,5,1,2,bias=True,padding_mode='reflect') 
        self.c7_a = nn.Conv2d(h_num,h_num,7,1,3,bias=True,padding_mode='reflect') 
        self.c1_b = nn.Conv2d(h_num,h_num,1,1,bias=True) 
        self.c3_b = nn.Conv2d(h_num,h_num,3,1,1,bias=True,padding_mode='reflect') 
        self.c5_b = nn.Conv2d(h_num,h_num,5,1,2,bias=True,padding_mode='reflect') 
        self.c7_b = nn.Conv2d(h_num,h_num,7,1,3,bias=True,padding_mode='reflect') 

        self.cat3 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.cat4 = nn.Conv2d(4,1,1,1,bias=True) 
        # self.cat1 = nn.Conv2d(number_f/2,number_f/2,1,1,bias=True) 

        # self.conv1 = nn.Conv2d(1,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv5 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv6 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv7 = nn.Conv2d(number_f*4,1,3,1,1,bias=True,padding_mode='reflect') 

        self.convdown2 = nn.Conv2d(2,2,2,2,bias=True) 

        self.conv11 = nn.Conv2d(2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv22 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv33 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv44 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv55 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv66 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        self.conv77 = nn.Conv2d(number_f*4,1,3,1,1,bias=True,padding_mode='reflect') 

        self.convdown4 = nn.Conv2d(2,2,4,4,bias=True) 

        # self.conv111 = nn.Conv2d(2,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv222 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv333 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv444 = nn.Conv2d(number_f,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv555 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv666 = nn.Conv2d(number_f*4,number_f,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv777 = nn.Conv2d(number_f*4,1,3,1,1,bias=True,padding_mode='reflect') 

        # self.res1 = resblock(number_f*2,number_f*2)
        # self.res2 = resblock(number_f*2,number_f*2)
        # self.res3 = resblock(number_f*2,number_f*2)
        # self.res4 = resblock(number_f*2,number_f*2)
        # self.res5 = resblock(number_f*2,number_f*2)

        # self.conv8 = nn.Conv2d(1,number_f*2,3,1,1,bias=True,padding_mode='reflect') 
        # self.conv9 = nn.Conv2d(number_f*2,1,3,1,1,bias=True,padding_mode='reflect') 

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        if pthfile is not None:
            # self.load_state_dict(torch.save(torch.load(pthfile), pthfile,_use_new_zipfile_serialization=False), strict = False)  # 训练所有数据后，保存网络的参数
            self.load_state_dict(torch.load(pthfile), strict = False)
        


    def forward(self, x, y):
        alph = 1000

        c = torch.cat((y,x),1)

        c1 = self.relu(self.conv11(c))
        c2 = self.relu(self.conv22(c1))
        c3 = self.relu(self.conv33(c2))
        c4 = self.relu(self.conv44(c3))
        c5 = self.relu(self.conv55(torch.cat([c1,c2,c3,c4],1)))
        c6 = self.relu(self.conv66(torch.cat([c1,c2,c3,c5],1)))
        c7 = self.relu(self.conv77(torch.cat([c1,c2,c3,c6],1)))
        # wxh = F.sigmoid(self.conv77(torch.cat([c1,c2,c3,c6],1)))

        c1a = self.c1_a(c7)
        # c1b = self.c1_b(cb)
        c3a = self.c3_a(c7)
        # c3b = self.c3_b(cb)
        c5a = self.c5_a(c7)
        # c5b = self.c5_b(cb)
        c7a = self.c7_a(c7)

        call = torch.cat((c1a,c3a,c5a,c7a),1)
        wxh = F.sigmoid(self.cat4(call)+c7)

        wyh = 1-wxh

        fuse = wxh*x+wyh*y

        return  wxh,wyh,wxh,wyh,c1a,c3a,c5a,c7a,fuse,wxh,wyh,wxh