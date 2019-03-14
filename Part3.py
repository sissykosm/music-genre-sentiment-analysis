
# coding: utf-8

# ## **Αναγνώριση Προτύπων - 3η Εργαστηριακή Άσκηση** ##
# 
# ## Αναγνώριση Είδους και Εξαγωγή Συναισθήματος από Μουσική ##

# * Χρυσούλα Κοσμά - 03114025
# * Λεωνίδας Αβδελάς - 03113182
# 
# 9ο Εξάμηνο ΣΗΜΜΥ ΕΜΠ

# Ακολουθούν τα **βήματα του κύριου μέρους (βήματα 10-11)** του 3ου Εργαστηρίου.
# 
# Αρχικά, κάνουμε import ορισμένες από τις βιβλιοθήκες που είναι απαραίτητες για την εκτέλεση των βημάτων της εργασίας.

# In[1]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gzip
import matplotlib.pyplot as plt
import librosa

from librosa import display
from librosa import beat


# In[2]:


print('Files in multitask_dataset folder')
print(os.listdir("../input/data/data/multitask_dataset"))
path = "../input/data/data/multitask_dataset/train_labels.txt"
f = open(path, 'r')
file_contents = f.read()
print('\nTrain_labels.txt in multitask_dataset')
print(file_contents)
f.close()


# In[3]:


path = "../input/data/data/multitask_dataset/train"
print('Files in multitask_dataset/train folder')
#print(os.listdir(path))


# In[4]:


import copy

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


# In[5]:


def torch_train_val_split(dataset, batch_train, batch_eval,val_size=.2, shuffle=True, seed=42):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)
    return train_loader, val_loader


# In[6]:


def read_spectrogram(spectrogram_file, chroma=True):
    with gzip.GzipFile(spectrogram_file, 'r') as f:
        spectrograms = np.load(f)
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T


# In[7]:


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


# In[8]:


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


# In[9]:


class MultitaskDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.train = False
        if train:
            self.train = True
            self.index = os.path.join(path, "{}_labels.txt".format(t))
            self.files, labels1, labels2, labels3 = self.get_files_labels(self.index, class_mapping)
        else:
            self.files = os.listdir(p)

        self.feats = [read_spectrogram(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        if self.train:
            self.label_transformer = LabelTransformer()
        
            if isinstance(labels1, (list, tuple)):
                self.labels1 = labels1
            if isinstance(labels2, (list, tuple)):
                self.labels2 = labels2
            if isinstance(labels3, (list, tuple)):
                self.labels3 = labels3

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
            #print(lines)
        files, labels1, labels2, labels3 = [], [], [], []
        for l in lines:
            chars = l[0].split(",")
            #print(chars[1])
            #print(chars[0]+'.fused.full.npy.gz')
            files.append(chars[0]+'.fused.full.npy.gz')
            labels1.append(float(chars[1]))
            labels2.append(float(chars[2]))
            labels3.append(float(chars[3]))
        return files, labels1, labels2, labels3

    def __getitem__(self, item):
        l = min(self.lengths[item], self.max_length)
        if self.train:
            return self.zero_pad_and_stack(self.feats[item]), self.labels1[item], self.labels2[item], self.labels3[item], l
        else:
            return self.zero_pad_and_stack(self.feats[item]), l
    def __len__(self):
        return len(self.files)        


# In[10]:


specs = MultitaskDataset('../input/data/data/multitask_dataset/', train=True, class_mapping=None, max_length=-1)
train_loader, val_loader = torch_train_val_split(specs, 45, 45, val_size=.33)


# In[11]:


import torch
from torch import nn
from torch.autograd import Variable
import math 

class CNN_d2(nn.Module):
    def __init__(self, num_classes,timesteps,num_features):
        super(CNN_d2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 10, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 15, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(0.01)
        cnn_out_dim = int(math.floor(timesteps/16)*math.floor(num_features/16))*20
        self.fc1 = nn.Linear(cnn_out_dim, math.floor(cnn_out_dim/10))
        self.fc2 = nn.Linear(math.floor(cnn_out_dim/10), math.floor(cnn_out_dim/100))
        self.fc3 = nn.Linear(math.floor(cnn_out_dim/100), 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# In[12]:


def eval_pred(features, model):
    output_tensor = model(features.unsqueeze_(1))
    batch_pred = output_tensor.view(1,-1)[0].detach()
    return batch_pred, output_tensor


# In[13]:


from scipy.stats import spearmanr

def train_val_loop(epochs,model,criterion,optimizer,task):
    
    for epoch in range(epochs):
        #train loop
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            features = torch.tensor(data[0]).float().cuda()
            labels = torch.tensor(data[task]).float().cuda()
            
            optimizer.zero_grad()           
            output = model(features.unsqueeze_(1))
            
            #print("OUT \n",output.permute(1,0))
            #print("Lab \n",labels)
            
            loss = criterion(output.permute(1,0),labels)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss

        num_batch_train = i+1

        val_loss = 0.0
        f1_val = 0.0
        
        #validation loop
        for j, data_val in enumerate(val_loader):
            features_val = torch.tensor(data_val[0]).float().cuda()
            labels_val = torch.tensor(data_val[task]).float().cuda()

            batch_pred, output_tensor = eval_pred(features_val, model)
            
            loss_val = criterion(output_tensor.permute(1,0),labels_val)
            val_loss = val_loss + loss_val
            #print("Batch Pred:", batch_pred)
            #print("Labels Val:", labels_val)
            #f1_val = f1_val + accuracy_score(labels_val.cpu(), batch_pred.cpu())
            #corr, _ = spearmanr(labels_val.cpu().squeeze().detach().numpy(), output_tensor.cpu().permute(1,0).squeeze().detach().numpy())
            #print(corr)
        num_batch_val = j+1    
        #f1_val = f1_val/num_batch_val
        
        print ('Epoch %d from %d, Train loss: %.4f' %(epoch + 1, epochs, train_loss/num_batch_train))
        print ('Epoch %d from %d, Validation loss: %.4f' %(epoch + 1, epochs, val_loss/num_batch_val))
        #print('Score in validation set is: %d %%' % (100 * f1_val))
        print('--------------------------------')
    return


# In[14]:


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

num_classes = 2
timesteps = 1293
num_features = 140

model1 = CNN_d2(num_classes,timesteps,num_features)
model1.cuda()

print('Training Loop for 2D CNN - Predictions for Valence')

epochs = 100
LR = 0.0008
weight_decay=0.0000005

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model1.parameters(), weight_decay=weight_decay, lr=LR)
train_val_loop(epochs,model1,criterion,optimizer,1)


# In[15]:



import warnings
warnings.filterwarnings('ignore')

num_classes = 2
timesteps = 1293
num_features = 140

model2 = CNN_d2(num_classes,timesteps,num_features)
model2.cuda()

print('Training Loop for 2D CNN - Predictions for Energy')

epochs = 100
LR = 0.0008
weight_decay=0.0000005

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model2.parameters(), weight_decay=0, lr=LR)
train_val_loop(epochs,model2,criterion,optimizer,2)


# In[16]:


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

num_classes = 2
timesteps = 1293
num_features = 140

model3 = CNN_d2(num_classes,timesteps,num_features)
model3.cuda()

print('Training Loop for 2D CNN - Predictions for Danceability')

epochs = 100
LR = 0.0008
weight_decay=0.0000005

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model3.parameters(), weight_decay=weight_decay, lr=LR)
train_val_loop(epochs,model3,criterion,optimizer,3)


# In[17]:


test_loader = DataLoader(MultitaskDataset('../input/data/data/multitask_dataset/', train=False, class_mapping=None, max_length=-1), batch_size=45)


# In[57]:


f = open('../results.txt', 'w')
f.write('Id.fused.full.npy.gz,valence,energy,danceability\n')
for j, data_test in enumerate(test_loader):
    #print(j)
    features = torch.tensor(data_test[0]).float().cuda()
    #labels_val = torch.tensor(data_test[task]).float().cuda()
    #print(data_test.shape[0])
    batch_pred1, _ = eval_pred(features, model1)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred2, _ = eval_pred(features, model2)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred3, _ = eval_pred(features, model3)
    for i in range(len(features)):
        #print(i)
        file = os.listdir('../input/data/data/multitask_dataset/test')[45*j+i]
        text = file + ',' + str(batch_pred1[i].cpu().numpy()) + ',' + str(batch_pred2[i].cpu().numpy()) + ',' + str(batch_pred3[i].cpu().numpy())
        f.write(text + '\n')
        #print(text)



    


# In[19]:


torch.save(model1, "../model01.pkl")
torch.save(model2, "../model02.pkl")
torch.save(model3, "../model03.pkl")


# In[58]:


cat ../results.txt


# In[21]:


validation_loader = DataLoader(MultitaskDataset('../input/data/data/multitask_dataset/', train=True, class_mapping=None, max_length=-1), batch_size=45)


# In[38]:


vale_pred = []
ene_pred = []
dance_pred = []
vale_labels = []
ene_labels = []
dance_labels = []
for j, data_test in enumerate(validation_loader):
    #print(j)
    features = torch.tensor(data_test[0]).float().cuda()
    vale_labels.append(data_test[1])
    ene_labels.append(data_test[2])
    dance_labels.append(data_test[3])
    #labels_val = torch.tensor(data_test[task]).float().cuda()
    #print(data_test.shape[0])
    batch_pred1, _ = eval_pred(features, model1)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred2, _ = eval_pred(features, model2)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred3, _ = eval_pred(features, model3)
    vale_pred.append(batch_pred1.cpu().numpy())
    ene_pred.append(batch_pred2.cpu().numpy())
    dance_pred.append(batch_pred3.cpu().numpy())


# In[42]:


flt_vale_labels = [item for sublist in vale_labels for item in sublist]
flt_ene_labels = [item for sublist in ene_labels for item in sublist]
flt_dance_labels = [item for sublist in dance_labels for item in sublist]
flt_vale_preds = [item for sublist in vale_pred for item in sublist]
flt_dance_preds = [item for sublist in dance_pred for item in sublist]
flt_ene_preds = [item for sublist in ene_pred for item in sublist]


# In[43]:


from scipy.stats import spearmanr

final_score_val , _ = spearmanr(flt_vale_labels, flt_vale_preds)
final_score_ene , _ = spearmanr(flt_ene_labels, flt_ene_preds)
final_score_dance, _ = spearmanr (flt_dance_labels, flt_dance_preds)

final_score = (final_score_val+final_score_ene+final_score_dance)/3

print("Final score is:", final_score)

