import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models
import spectral
import clip
# np.random.seed(1337)

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 80) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 10)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=10)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()#创建文件

# load source domain data set
with open(os.path.join('/data/hulei/2/Cross_modal/datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys()) #dict_keys(['data', 'Labels', 'set'])
print(source_imdb['Labels']) #(42776,)

# process source domain data set
data_train = source_imdb['data'] # (42776, 9, 9, 103)
labels_train = source_imdb['Labels'] # (42776,)
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,8]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))#9
print(data.keys()) #dict_keys([1, 2, 5, 0, 7, 3, 6, 8, 4])
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))#9

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,103）-> (103,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape) # (42776, 9, 9, 103)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #
print(source_imdb['data'].shape) # (9, 9, 103, 42776)
print(source_imdb['Labels'].shape)#(42776,)
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
def load_data_ip(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    data_all = image_data['imggt']
    label = label_data['imggt']
    gt = label.reshape(np.prod(label.shape[:2]), )#(207400,)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  #(207400,103)
    print(data.shape) #
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])#(610,340,103)

    return data_scaler, gt
test_data = '/data/hulei/Datasets/Tea/Tea.mat'
test_label = '/data/hulei/Datasets/Tea/Tea_gt.mat'
Data_Band_Scaler, GroundTruth = load_data_ip(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (145, 145, 220)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)#(435,435,220)
    gt1=GroundTruth.reshape(512,348)
    groundtruth = utils.flip(gt1) #(435,435)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]#(153,153)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]#(153,153,220)

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)#10249
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 16*5;520
    print('the number of test_indices:', len(test_indices))  # 10169;9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.flip_augmentation(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain

def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = next(iter(train_loader))
    print('train labels:', train_labels) #80
    print('size of train datas:', train_datas.shape) # size of train datas:torch.Size([80, 220, 9, 9]); torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())#dict_keys(['data', 'Labels', 'set'])
    print(imdb_da_train['data'].shape)  # (9, 9, 220, 3200); (9, 9, 100, 225)
    print(imdb_da_train['Labels'].shape) #(3200,)
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (3200, 220, 9, 9);(9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape) #(3200, 220, 9, 9)
    target_da_labels = imdb_da_train['Labels']  #(3200,);(1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 220, 3200);(9, 9, 100, 225)
    print(imdb_da_train['Labels'].shape)#(3200,)
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


# model
def conv3x3x3(in_channels, out_channels,kernel_size=(51, 3, 3), stride=1,padding=0):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride,padding=padding,bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=False)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.block1(x) #(1,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)
        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)
        x = self.conv(x) #(1,32,5,1,1)
        x = x.view(x.shape[0],-1) #(1,160)
        # y = self.classifier(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)#128->100

    def forward(self, x, domain='source'):  # x
        # print(x.shape)
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        # print(x.shape)#torch.Size([45, 100, 9, 9])
        feature = self.feature_encoder(x)  # (45, 64)
        # print((feature.shape))
        output = self.classifier(feature)
        return feature, output
    
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm3d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x
class Att(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1

    
class ConvBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=1):
        super(ConvBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
        super(ConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x


class GhostModule3D(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule3D, self).__init__()
        self.oup = oup 
        init_channels = math.ceil(oup / ratio) 
        new_channels = init_channels*(ratio-1) 

        self.primary_conv = nn.Sequential(
            nn.Conv3d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:, :self.oup, :, :]
    
class HyperCLR(nn.Module):
    def __init__(self):
        # 调用Module的初始化
        super(HyperCLR, self).__init__()
        # self.channel=channel
        # self.output_units=output_units
        # self.windowSize=windowSize
        self.conv1 = conv3x3x3(in_channels=1,out_channels= 8,kernel_size=(3,3,3),stride=1,padding=1)
        self.ghost_cheaper3d_1 = GhostModule3D(inp=8,oup=16,relu=True)
        self.conv11 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(1,1,1), stride=1,padding=0)
        self.bn1 = nn.BatchNorm3d(num_features=16)
        self.Att1 = Att(16)
        
        # self.AP1 = nn.AvgPool3d(3, stride=2)
        self.AP1 = nn.AvgPool3d(kernel_size=(4,4,4))
        
        self.conv2 = conv3x3x3(in_channels=16,out_channels=16,kernel_size=(1,1,1),stride=1,padding=0)
        self.ghost_cheaper3d_2 = GhostModule3D(inp=16,oup=32,relu=True)
        self.conv21 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1,1,1), stride=1,padding=0)
        self.bn2 = nn.BatchNorm3d(num_features=32)
        self.Att2 = Att(32)

        self.AP2 = nn.AvgPool3d(kernel_size=(3,2,2))
        # self.projector = nn.Sequential(
        #     # nn.Linear(256, 128),
        #     # nn.ReLU(),
        #     nn.Linear(768,512),
        # )
        # self.fc=nn.Linear(256,128)
        # self.relu1=nn.ReLU()
        self.fc2=nn.Sequential(
            nn.Linear(1600, CLASS_NUM))
        
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)#128->100
    def forward(self, x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        x = x.unsqueeze(1)
        x0 = self.conv1(x) #(-1,8,18,13,13)
        # x1 = self.Ar1(x0)
        x1 = self.ghost_cheaper3d_1(x0)
        x1 = self.Att1(x1)
        x12 = self.conv11(x0)
        x12 = self.bn1(x12)
        x13 = x1+x12
        AP1 = self.AP1(x13)
        L = AP1.view([AP1.shape[0], -1])
        
        h = L
        # c=self.fc(L)
        # c=self.relu1(c)
        # z=self.fc2(L)
        
        return h  

# from torchsummary import summary
# model=HyperCLR()
# model=model.cuda()
# summary(model,(1,100,21,21))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

# def euclidean_metric(a, b):
#     n = a.shape[0]
#     m = b.shape[0]
#     a = a.unsqueeze(1).expand(n, m, -1)
#     b = b.unsqueeze(0).expand(n, m, -1)
#     logits = -((a - b)**2).sum(dim=2)
#     return logits

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations={}
    class_precision_matrices={}
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        class_features = torch.index_select(context_features, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    return class_representations,class_precision_matrices

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()
#clip
#device = "cuda" if torch.cuda.is_available() else "cpu"
class LogitHead(nn.Module):
    def __init__(self, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = nn.Linear(768, 768, bias=False)
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        # x = x * self.logit_scale.exp()
        return x

class LogitHead1(nn.Module):
    def __init__(self, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = nn.Linear(1600, 768, bias=False)
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        # x = x * self.logit_scale.exp()
        return x

model_clip, preprocess = clip.load("ViT-L/14")
#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

label_values1 = ["Water", "Bare soil school", "Bare soil park", "Bare soil farmland", "Natural plants", "Weeds in farmland", "Forest", "Grass", "Rice field grown", "Rice field first stage", "Row crops", "Plastic house", "Manmade non-dark", "Manmade dark", "Manmade blue", "Manmade red", "Manmade grass", "Asphalt", "Paved ground"]
label_values2 = ["Masson pine", "Bamboo forest", "Tea plant", "Reed", "Rice paddy", "Sweet potato", "Caraway", "Weed", "Water", "Building and road"]

class MMSD(nn.Module):
    def __init__(self):
        super(MMSD, self).__init__()

    def _mix_rbf_mmsd(self, X, Y, sigmas=(1,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = torch.tensor(K_XX.size(0), dtype=torch.float32)
        n = torch.tensor(K_YY.size(0), dtype=torch.float32)
        C_K_XX = torch.pow(K_XX, 2)
        C_K_YY = torch.pow(K_YY, 2)
        C_K_XY = torch.pow(K_XY, 2)
        if biased:
            mmsd = (torch.sum(C_K_XX) / (m * m) + torch.sum(C_K_YY) / (n * n)
            - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return mmsd

    def forward(self, X1, X2, bandwidths=[3]):
        kernel_loss = self._mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
        return kernel_loss

MMSDloss=MMSD()

#domain_classifer
class DC(nn.Module):
    def __init__(self,logit_scale=float(np.log(1 / 0.07))):
        super(DC, self).__init__()
        self.logit_scale = logit_scale
        self.logit_scale = nn.Parameter(torch.FloatTensor([logit_scale]).cuda())
        self.domain = nn.Linear(768, CLASS_NUM) 

    def forward(self, x):
        x = F.normalize(x, dim=1)
        domain_y = self.domain(x)
        domain_y = domain_y * self.logit_scale.exp()
        return domain_y
    
class Domain_Occ_loss(nn.Module):
    def __init__(self):
        super(Domain_Occ_loss,self).__init__()

    def forward(self,p1,p2):#(64,1)  (64,1)
        loss = - torch.mean(torch.log(p1 + 1e-6))
        loss -= torch.mean(torch.log(p2 + 1e-6))

        return loss
DSH_loss = Domain_Occ_loss()
# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
teslr=0.01

seeds = [1336]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = HyperCLR()
    logit_head = LogitHead()
    logit_head1 = LogitHead1()
    Domian_Classifier = DC()

    feature_encoder.apply(weights_init)
    logit_head.apply(weights_init)
    logit_head1.apply(weights_init)
    Domian_Classifier.apply(weights_init)

    feature_encoder.cuda()
    model_clip.cuda()
    logit_head.cuda()
    logit_head1.cuda()
    Domian_Classifier.cuda()

    feature_encoder.train()
    logit_head.train()
    logit_head1.train()
    Domian_Classifier.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=teslr)
    logit_head_optim = torch.optim.Adam(logit_head.parameters(), lr=teslr)
    logit_head_optim1 = torch.optim.Adam(logit_head1.parameters(), lr=teslr)
    Domian_Classifier_optim = torch.optim.Adam(Domian_Classifier.parameters(), lr=teslr)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(10000):  # EPISODE = 10000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            # source_iter = iter(source_loader)
            source_data, source_label = next(iter(source_loader))

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            # target_iter = iter(target_loader)
            target_data, target_label =next(iter(target_loader))

        lambd = -4 / (np.sqrt(episode / (EPISODE - episode + 1)) + 1) + 4 
        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 16， 1，19
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = next(iter(support_dataloader))  # (5, 100, 9, 9)
            #print(support_labels)[0-8]
            #print(supports.shape)[9,128,9,9]
            querys, query_labels = next(iter(query_dataloader))  # (75,100,9,9)
            
            # calculate features
            support_features = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_features= feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])
            target_features= feature_encoder(target_data.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            #print(support_features.shape)#[9,512]
            #print(query_features.shape)#[171,768]
            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            #print(logits)

            f_loss = crossEntropy(logits, query_labels.long().cuda())

            # clip features
            for param in model_clip.parameters():
                   param.requires_grad = False
            #print(text)
            #print(text.shape)#[9,77]
            text = torch.cat([clip.tokenize(f'An image of {label_values1[k]}').to(k) for k in support_labels])
            support_textfeature=model_clip.encode_text(text.cuda())
            #print(support_textfeature)
            #print(support_textfeature.shape)#[9,512]
            # logits1 = euclidean_metric(query_features, support_textfeature)
            # #print(logits)
            # f_loss1 = crossEntropy(logits1, query_labels.long().cuda())

            # feature = torch.cat([support_features, support_textfeature], dim=0)
            # # print(feature.shape)
            # label = torch.cat([support_labels, support_labels], dim=0)
            support_textfeature = logit_head(support_textfeature.float())
            support_features = logit_head1(support_features)
            # loss_clip = crossEntropy(logit, label.long().cuda())

            support_features = support_features / support_features.norm(dim=1, keepdim=True)
            support_textfeature = support_textfeature / support_textfeature.norm(dim=1, keepdim=True)
            support_textfeature=support_textfeature.float()
            logits_per_image = support_features @ support_textfeature.T
            logits_per_text = support_textfeature @ support_features.T
            loss_img = crossEntropy(logits_per_image, support_labels.long().cuda())
            loss_text = crossEntropy(logits_per_text,
            support_labels.long().cuda())
            loss_clip = (loss_img + loss_text)/2

            # outputs_sou = torch.cat((so, qo), dim=0)
            # # outputs_tar = target_outputs
            # MMSD_loss = MMSDloss(outputs_sou, taro)
            # text_q = torch.cat([clip.tokenize(f'An image of {label_values1[k]}').to(k) for k in query_labels])
            # q_textfeature=model_clip.encode_text(text_q.cuda())
            # q_textfeature=logit_head(q_textfeature.float())
            # q_textfeature = q_textfeature / q_textfeature.norm(dim=1, keepdim=True)
            # q_textfeature=q_textfeature.float()
            # query_features=logit_head1(query_features)
            # query_features = query_features / query_features.norm(dim=1, keepdim=True)

            # sou_f = torch.cat((support_features, query_features), dim=0)
            # sou_tf = torch.cat((support_textfeature, q_textfeature), dim=0)

            S_F = 0.5*support_features + 0.5*support_textfeature
            S_C = Domian_Classifier(S_F)

            # target_features = logit_head1(target_features)
            # target_features = target_features/target_features.norm(dim=1, keepdim=True)

            # tar_text = torch.cat([clip.tokenize(f'An image of {label_values2[k]}').to(k) for k in target_label])
            # tar_textfeatures=model_clip.encode_text(tar_text.cuda())

            # tar_textfeatures = logit_head(tar_textfeatures.float())
            # tar_textfeatures = tar_textfeatures/tar_textfeatures.norm(dim=1, keepdim=True)
            # tar_textfeatures = tar_textfeatures.float()

            # T_F = 0.5*target_features + 0.5*tar_textfeatures
            # T_C = Domian_Classifier(T_F)

            Similar_loss = crossEntropy(S_C,support_labels.long().cuda())                                   
            # total_loss = fsl_loss + domain_loss
            loss = f_loss+loss_clip+Similar_loss
            # Update parameters
            feature_encoder.zero_grad()
            logit_head.zero_grad()
            logit_head1.zero_grad()
            Domian_Classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            logit_head_optim.step()
            logit_head_optim1.step()
            Domian_Classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = next(iter(support_dataloader))   # (5, 100, 9, 9)
            print('supports',supports.shape,support_labels.shape)

            
            querys, query_labels = next(iter(query_dataloader))   # (75,100,9,9)
            print('querys',querys.shape,query_labels.shape)

            # calculate features
            support_features= feature_encoder(supports.cuda(),  domain='target')  # torch.Size([409, 32, 7, 3, 3])
            print('support_features',support_features.shape)
            query_features= feature_encoder(querys.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_features= feature_encoder(source_data.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            
            f_loss = crossEntropy(logits, query_labels.long().cuda())

            # clip features
            #print(text)
            #print(text.shape)
            for param in model_clip.parameters():
                   param.requires_grad = False
            text = torch.cat([clip.tokenize(f'An image of {label_values2[k]}').to(k) for k in support_labels])
            support_textfeature=model_clip.encode_text(text.cuda())
            print('support_textfeature',support_textfeature.shape)
            #print(support_textfeature)
            #print(support_textfeature.shape)

            # logits1 = euclidean_metric(query_features, support_textfeature)
            # #print(logits)
            # f_loss1 = crossEntropy(logits1, query_labels.long().cuda())
            support_textfeature = logit_head(support_textfeature.float())
            print('support_textfeature',support_textfeature.shape)
            support_features = logit_head1(support_features)
            print('support_features',support_features.shape)
            # feature = torch.cat([support_features, support_textfeature], dim=0)
            # # print(feature.shape)
            # label = torch.cat([support_labels, support_labels], dim=0)
            # logit = logit_head(feature)
            # loss_clip = crossEntropy(logit, label.long().cuda())

            support_features = support_features / support_features.norm(dim=1, keepdim=True)
            print('support_features',support_features.shape)
            support_textfeature = support_textfeature / support_textfeature.norm(dim=1, keepdim=True)
            print('support_textfeature',support_textfeature.shape)
            support_textfeature=support_textfeature.float()
            logits_per_image = support_features @ support_textfeature.T
            print('logits_per_image',logits_per_image.shape)
            logits_per_text = support_textfeature @ support_features.T
            print('logits_per_text',logits_per_text.shape)
            loss_img = crossEntropy(logits_per_image, support_labels.long().cuda())
            print('loss_imgt',loss_img.shape)
            loss_text = crossEntropy(logits_per_text, support_labels.long().cuda())
            print('loss_text',loss_text.shape)
            loss_clip = (loss_img + loss_text)/2


            # text_q = torch.cat([clip.tokenize(f'An image of {label_values2[k]}').to(k) for k in query_labels])
            # q_textfeature=model_clip.encode_text(text_q.cuda())
            # q_textfeature=logit_head(q_textfeature.float())
            # q_textfeature = q_textfeature / q_textfeature.norm(dim=1, keepdim=True)
            # q_textfeature=q_textfeature.float()
            # query_features=logit_head1(query_features)
            # query_features = query_features / query_features.norm(dim=1, keepdim=True)

            # sou_f = torch.cat((support_features, query_features), dim=0)
            # sou_tf = torch.cat((support_textfeature, q_textfeature), dim=0)

            S_F = 0.5*support_features + 0.5*support_textfeature
            S_C = Domian_Classifier(S_F)
            print('S_C',S_C.shape)

            # source_features = logit_head1(source_features)
            # source_features = source_features/source_features.norm(dim=1, keepdim=True)

            # sou_text = torch.cat([clip.tokenize(f'An image of {label_values1[k]}').to(k) for k in source_label])
            # sou_textfeatures=model_clip.encode_text(sou_text.cuda())

            # sou_textfeatures = logit_head(sou_textfeatures.float())
            # sou_textfeatures = sou_textfeatures/sou_textfeatures.norm(dim=1, keepdim=True)
            # sou_textfeatures = sou_textfeatures.float()

            # T_F = 0.5*source_features + 0.5*sou_textfeatures
            # T_C = Domian_Classifier(T_F)

            Similar_loss = crossEntropy(S_C,support_labels.long().cuda())
            print('Similar_loss',Similar_loss.shape)

                                   
            # total_loss = fsl_loss + domain_loss
            loss = f_loss+loss_clip+Similar_loss
            # Update parameters
            feature_encoder.zero_grad()
            logit_head.zero_grad()
            logit_head1.zero_grad()
            Domian_Classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            logit_head_optim.step()
            logit_head_optim1.step()
            Domian_Classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  , fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                
                                                                                                                f_loss.item(),
                                                                                                                total_hit / total_num,
                                                                                                                loss.item()))

        if (episode + 1) % 100 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = next(iter(train_loader))
            train_features = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            print(train_features.shape)
            if TEST_LSAMPLE_NUM_PER_CLASS > 1:
                train_proto = train_features.reshape(CLASS_NUM, TEST_LSAMPLE_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                train_proto = train_features


            print('train_features_size:',train_features.shape)
            embed_fea = np.empty((0,1600), dtype=np.int64)    
            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            # train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            # KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            # KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
          
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                logits = euclidean_metric(test_features, train_proto)
                predict_labels = torch.argmax(logits, dim=1).cpu()

                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                embed_fea = np.append(embed_fea,test_features.cpu().detach().numpy(),axis=0)

                

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str("/data/hulei/2/Cross_modal/T-SNE/TeDFL" + "TEA_all" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                best_emd = embed_fea
                labels_emd = labels
                # np.save('/data/hulei/2/Cross_modal/T-SNE/tsne_data/fea_tsne_TEA'+'.npy',best_emd)
                # np.save('/data/hulei/2/Cross_modal/T-SNE/tsne_data/label_tsne_TEA'+'.npy',labels_emd)

            print('best episode:[{}], best accuracy={}, confusion_matrix={}'.format(best_episdoe + 1, last_accuracy, C))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


# best_iDataset = 0
# for i in range(len(acc)):
#     print('{}:{}'.format(i, acc[i]))
#     if acc[i] > acc[best_iDataset]:
#         best_iDataset = i
# print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

# for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
#     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

# hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# for i in range(best_G.shape[0]):
#     for j in range(best_G.shape[1]):
#         if best_G[i][j] == 0:
#             hsi_pic[i, j, :] = [0.6, 0.8, 1]
#         if best_G[i][j] == 1:
#             hsi_pic[i, j, :] = [0, 0, 1]
#         if best_G[i][j] == 2:
#             hsi_pic[i, j, :] = [0, 1, 0]
#         if best_G[i][j] == 3:
#             hsi_pic[i, j, :] = [0, 1, 1]
#         if best_G[i][j] == 4:
#             hsi_pic[i, j, :] = [1, 0, 0]
#         if best_G[i][j] == 5:
#             hsi_pic[i, j, :] = [1, 0, 1]
#         if best_G[i][j] == 6:
#             hsi_pic[i, j, :] = [1, 1, 0]
#         if best_G[i][j] == 7:
#             hsi_pic[i, j, :] = [0.5, 0.5, 1]
#         if best_G[i][j] == 8:
#             hsi_pic[i, j, :] = [0.65, 0.35, 1]
#         if best_G[i][j] == 9:
#             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
#         if best_G[i][j] == 10:
#             hsi_pic[i, j, :] = [0.75, 1, 0.5]
#         if best_G[i][j] == 11:
#             hsi_pic[i, j, :] = [0.5, 1, 0.65]
#         if best_G[i][j] == 12:
#             hsi_pic[i, j, :] = [0.65, 0.65, 0]
#         if best_G[i][j] == 13:
#             hsi_pic[i, j, :] = [0.75, 1, 0.65]
#         if best_G[i][j] == 14:
#             hsi_pic[i, j, :] = [0, 0, 0.5]
#         if best_G[i][j] == 15:
#             hsi_pic[i, j, :] = [0, 1, 0.75]
#         if best_G[i][j] == 16:
#             hsi_pic[i, j, :] = [0.5, 0.75, 1]
#         if best_G[i][j] == 17:
#             hsi_pic[i, j, :] = [0.5, 0.5, 0.5]
#         if best_G[i][j] == 18:
#             hsi_pic[i, j, :] = [0, 0.5, 0.5]
#         if best_G[i][j] == 19:
#             hsi_pic[i, j, :] = [1, 0.5, 0]
#         if best_G[i][j] == 20:
#             hsi_pic[i, j, :] = [0.6, 0.2, 0]


# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "/home/hulei/2/Cross_modal/LR/Tea_Adam_lr0.00001_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))

file_name = r"/data/hulei/2/Cross_modal/T-SNE/Tea_Adam_lr0.01_all.txt"
with open(file_name, 'w') as x_file:
    x_file.write('train time per DataSet(s): '+"{:.5f}".format(train_end-train_start))
    x_file.write('\n')
    x_file.write('test time per DataSet(s):'+"{:.5f}".format(test_end-train_end))
    x_file.write('\n')
    x_file.write('average OA:'+"{:.5f}".format( OAMean) + " +- " + "{:.5f}".format( OAStd))
    x_file.write('\n')
    x_file.write('{} average AA:'+"{:.5f}".format(100 * AAMean) + " +- " + "{:.5f}".format(100 * AAStd))
    x_file.write('\n')
    x_file.write('{} average Kapppa:'+"{:.5f}".format(100 *kMean) + " +- " + "{:.5f}".format(100 *kStd))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy:'+format(k))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%):'+format(acc))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'+format(A))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} confusion:'+format(C.astype(str)))

