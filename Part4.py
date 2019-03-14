
# coding: utf-8

# ## **Αναγνώριση Προτύπων - 3η Εργαστηριακή Άσκηση** ##
# 
# ## Αναγνώριση Είδους και Εξαγωγή Συναισθήματος από Μουσική ##

# * Χρυσούλα Κοσμά - 03114025
# * Λεωνίδας Αβδελάς - 03113182
# 
# 9ο Εξάμηνο ΣΗΜΜΥ ΕΜΠ

# # Βήμα 11#

# Τα συμπεράσματα του [5] είναι ότι η μεταφορά βαρών ενός νευρωνικού δικτύου (Α) σε ένα άλλο (Β) επιφέρει καλύτερα αποτελέσματα από ότι αν χρησιμοποιούσαμε τυχαία αρχικοποιημένα βάρη. Ακόμα το σημείο που θα κόψουμε το νευρωνικό Α δεν επηρεάζει κατα πολύ την βελτίωση της απόδοσης, αν αυτά τα βάρη τα προσαρμόσουμε μετά στο νέο dataset (fine-tuning).

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


import copy

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


# In[3]:


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


# In[4]:


def read_spectrogram(spectrogram_file, chroma=True):
    with gzip.GzipFile(spectrogram_file, 'r') as f:
        spectrograms = np.load(f)
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T


# In[5]:


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


# In[6]:


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


# In[7]:


class SpectrogramDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            files.append(l[0])
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)


# In[8]:


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


# In[9]:


specs = SpectrogramDataset('../input/data/data/fma_genre_spectrograms/', train=True, class_mapping=None, max_length=-1)
train_loader, val_loader = torch_train_val_split(specs, 45, 45, val_size=.33)


# In[10]:


import torch
from torch import nn
from torch.autograd import Variable
import math 

class CNN_d2(nn.Module):
    def __init__(self, num_classes,timesteps,num_features):
        super(CNN_d2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.dropout = nn.Dropout(0.0001)
        cnn_out_dim = int(math.floor(timesteps/16)*math.floor(num_features/16)*16)
        self.fc1 = nn.Linear(cnn_out_dim, math.floor(cnn_out_dim/100))
        self.fc2 = nn.Linear(math.floor(cnn_out_dim/100), num_classes)
        
    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# In[11]:


def eval_pred(features):
    output_tensor = model(features.unsqueeze_(1))
    batch_pred = torch.argmax(output_tensor.data, dim=1)
    return batch_pred, output_tensor


# In[12]:


# Early stopping

class EarlyStopping(object):
    
    """
    EarlyStopping can be used to stop te training if no improvement after a given number of events
    
    Args: 
        patience(int):
            Number of events to wait if no improvement and then stop the training
        
        mode(string):
            There are two modes:
                min, for looking for minumums
                max, for looking for maximums
                
        min_delta(float):
            The threshold of improvement
            
        percentage(boolean):
            Defines whether min_delta is a percentage or an absolute number
    """
    
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0 # counter of no events
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    """
    Returns True if the Early Stopping has to be enforced, otherwise returns False.
    """
    
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


# In[13]:


def train_val_loop(epochs,model,criterion,optimizer,earlystopping):
    
    for epoch in range(epochs):
        #train loop
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            features = torch.tensor(data[0]).float().cuda()
            labels = torch.tensor(data[1]).long().cuda()

            optimizer.zero_grad()           
            output = model(features.unsqueeze_(1))

            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss

        num_batch_train = i+1

        val_loss = 0.0
        f1_val = 0.0

        #validation loop
        for j, data_val in enumerate(val_loader):
            features_val = torch.tensor(data_val[0]).float().cuda()
            labels_val = torch.tensor(data_val[1]).long().cuda()

            batch_pred, output_tensor = eval_pred(features_val)

            loss_val = criterion(output_tensor,labels_val)
            val_loss = val_loss + loss_val

            f1_val = f1_val + accuracy_score(labels_val.cpu(), batch_pred.cpu())

        num_batch_val = j+1    
        f1_val = f1_val/num_batch_val

        print ('Epoch %d from %d, Train loss: %.2f' %(epoch + 1, epochs, train_loss/num_batch_train))
        print ('Epoch %d from %d, Validation loss: %.2f' %(epoch + 1, epochs, val_loss/num_batch_val))
        print('Score in validation set is: %d %%' % (100 * f1_val))
        print('--------------------------------')
        
        if(earlystopping.step(f1_val) is True):
            print('Early stopping the training cycle on epoch %d .' %(epoch+1))
            break
    return


# Εκπαιδεύουμε το 2D CNN του βήματος 9 για 100 εποχές με Early Stopping.

# In[14]:


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

num_classes = 20
timesteps = 1293
num_features = 140

model = CNN_d2(num_classes,timesteps,num_features)
model.cuda()

print('Training Loop for 2D CNN')

epochs = 100
LR = 0.008
weight_decay=0.05

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay, lr=LR)
earlystopping = EarlyStopping(mode='max', min_delta=0.01, patience=8)
train_val_loop(epochs,model,criterion,optimizer,earlystopping)


# In[15]:


torch.save(model, "../model.pkl")
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# In[16]:


# Model class must be defined somewhere
model3 = model2 = model1 = torch.load("../model.pkl")
model1.eval()
model2.eval()
model3.eval()


# Αλλάζουμε τις παραμέτρους εισόδου και εξόδου στις Linear συναρτήσεις ώστε να αντιστοιχούν στις εξόδους του multitask dataset. 

# In[17]:


specs = MultitaskDataset('../input/data/data/multitask_dataset/', train=True, class_mapping=None, max_length=-1)
train_loader, val_loader = torch_train_val_split(specs, 45, 45, val_size=.33)
test_loader = DataLoader(MultitaskDataset('../input/data/data/multitask_dataset/', train=False, class_mapping=None, max_length=-1), batch_size=45)


# In[18]:


def eval_pred_multi(features, model):
    output_tensor = model(features.unsqueeze_(1))
    batch_pred = output_tensor.view(1,-1)[0].detach()
    return batch_pred, output_tensor


# In[19]:


from scipy.stats import spearmanr

def train_val_loop_multi(epochs,model,criterion,optimizer,task):
    
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

            batch_pred, output_tensor = eval_pred_multi(features_val, model)
            
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


# In[20]:


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

num_classes = 2
timesteps = 1293
num_features = 140


cnn_out_dim = int(math.floor(timesteps/16)*math.floor(num_features/16))*16
model1.fc1 = nn.Linear(cnn_out_dim, math.floor(cnn_out_dim/100))
model1.fc2 = nn.Linear(math.floor(cnn_out_dim/100), 1)
model1.cuda()

epochs = 20
LR = 0.0008
weight_decay=0.0000005
print('Training Loop for 2D CNN - Predictions for Valence')

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model1.parameters(), weight_decay=weight_decay, lr=LR)
train_val_loop_multi(epochs,model1,criterion,optimizer,1)


# In[21]:


model3.fc1 = nn.Linear(cnn_out_dim, math.floor(cnn_out_dim/100))
model3.fc2 = nn.Linear(math.floor(cnn_out_dim/100), 1)
model3.cuda()

print('Training Loop for 2D CNN - Predictions for Danceability')

epochs = 20
LR = 0.0008
weight_decay=0.0000005

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model3.parameters(), weight_decay=weight_decay, lr=LR)
train_val_loop_multi(epochs,model3,criterion,optimizer,3)


# In[22]:


model2.fc1 = nn.Linear(cnn_out_dim, math.floor(cnn_out_dim/100))
model2.fc2 = nn.Linear(math.floor(cnn_out_dim/100), 1)
model2.cuda()

print('Training Loop for 2D CNN - Predictions for Energy')

epochs = 20
LR = 0.0008
weight_decay=0.0000005

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model2.parameters(), weight_decay=weight_decay, lr=LR)
train_val_loop_multi(epochs,model2,criterion,optimizer,2)


# In[ ]:


f = open('../results_transfer.txt', 'w')
f.write('Id.fused.full.npy.gz,valence,energy,danceability')
for j, data_test in enumerate(test_loader):
    #print(j)
    features = torch.tensor(data_test[0]).float().cuda()
    #labels_val = torch.tensor(data_test[task]).float().cuda()
    #print(data_test)
    batch_pred1, _ = eval_pred_multi(features, model1)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred2, _ = eval_pred_multi(features, model2)
    features = torch.tensor(data_test[0]).float().cuda()
    batch_pred3, _ = eval_pred_multi(features, model3)
    for i in range(len(features)):
        file = os.listdir('../input/data/data/multitask_dataset/test')[45*j+i]
        f.write(file + ',' + str(batch_pred1[i].cpu().numpy()) + ',' + str(batch_pred2[i].cpu().numpy()) + ',' + str(batch_pred3[i].cpu().numpy()))
f.close()

