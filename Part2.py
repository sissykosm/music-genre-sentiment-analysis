#!/usr/bin/env python
# coding: utf-8

# ## **Αναγνώριση Προτύπων - 3η Εργαστηριακή Άσκηση** ##
# 
# ## Αναγνώριση Είδους και Εξαγωγή Συναισθήματος από Μουσική ##

# Χρυσούλα Κοσμά - 03114025
# 
# Λεωνίδας Αβδελάς - 03113182
# 
# 9ο Εξάμηνο ΣΗΜΜΥ ΕΜΠ
# 

# Ακολουθούν τα **βήματα του κύριου μέρους (βήματα 9-12)** του 3ου Εργαστηρίου.
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


import copy

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


# Επαναλαμβάνουμε το **Βήμα 4 - Φόρτωση Δεδομένων** που βρίσκεται στο notebook της προπαρασκευής, για να δημιουργήσουμε τους Data Loaders που θα χρειαστούν για τα μοντέλα που υλοποιούμε στα υπόλοιπα βήματα.

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


# ## **Βήμα 9 - 2D CNN** ##

# Αρχικά, δημιουργούμε τους train_loader, val_loader, test_loader που θα χρησιμοποιήσουμε για την εκπαίδευση και την αξιολόγηση του 2D CNN. Επιλέγουμε για το λόγο αυτό τα fma_genre_spectrograms που βρίσκονται στο φάκελο '../input/data/data/fma_genre_spectrograms/', καθώς και το κατάλληλο batch_size. 

# In[8]:


specs = SpectrogramDataset('../input/data/data/fma_genre_spectrograms/', train=True, class_mapping=None, max_length=-1)
train_loader, val_loader = torch_train_val_split(specs, 45, 45, val_size=.33)
test_loader = DataLoader(SpectrogramDataset('../input/data/data/fma_genre_spectrograms/', train=False, class_mapping=None, max_length=-1))


# Xρησιμοποιώντας τα spectrograms της βάσης δεδομένων FMA εκπαιδεύουμε ένα 2D CNN δίκτυο, το οποίο θα δέχεται ως είσοδο τα
# σπεκτρογράμματα (train set) και θα προβλέπει τις 20 διαφορετικές κλάσεις (μουσικά είδη) του dataset. Για την εκπαίδευση χρησιμοποιούμε και validation set και ενεργοποιούμε τη GPU (χρήση cuda()).
# Συγκεκριμένα, υλοποιήσαμε:
# 1. Την κλάση **CNN_d2** η οποία αποτελεί ένα 2D CNN με 4 layers που επεξεργάζεται το σπεκτρόγραμμα σαν μονοκάναλη εικόνα, και κάθε layer πραγματοποιεί με αυτή τη σειρά: 2D convolution, Batch normalization, ReLU activation, Max pooling. To 1o nn.Conv2dr έχει in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2, τo 2o nn.Conv2d έχει in_channels=2, out_channels=4, kernel_size=5, stride=1, padding=2, τo 3o nn.Conv2d έχει in_channels=4, out_channels=6, kernel_size=5, stride=1, padding=2 και τo 4o nn.Conv2d έχει in_channels=6, out_channels=8, kernel_size=5, stride=1, padding=2. Όλα τα nn.MaxPool2d έχουν kernel_size=2, stride=2. Μετά τα 4 con2d layers περνάμε την έξοδο από ένα nn.Linear για τον μετασχησμό των προβλέψεων των 20 κλάσεων. Τα παραπάνω layers εφαρμόζονται με τη σειρά στην είσοδο του νευρωνικού στη συνάρτηση forward από όπου επειστρέφεται και το output στη μορφή batch_size x num_classes (πλήθος κλάσεων).
# 2. Τη συνάρτηση **eval_pred**, η οποία παράγει από την έξοδο του νευρωνικού τις προβλέψεις για τις ετικέτες κάθε δείγματος του batch. Για το σκοπό αυτό  στο output του νευρωνικού (που είναι της μορφής μέγεθος batch επί 20 κλάσεις) εφαρμόζει την torch.argmax ανά γραμμή (δηλαδή δείγμα) για να επιστραφεί η ετικέτα με την μεγαλύτερη τιμή. Επιστρέφονται οι προβλέψεις για όλα τα δείγματα, καθώς και η έξοδος του νευρωνικού.
# 3. Τη συνάρτηση που περιλαμβάνει το **train loop - υπολογισμό train/validation loss ανά εποχή**. Για το πλήθος των εποχών που έχουμε ορίσει,  το μοντέλο, τον optimizer και loss function που έχουμε επιλέξει, σε κάθε μια από αυτές παίρνουμε ένα batch από τον train loader που έχουμε δημιουργήσει καλούμε το μοντέλο και βρικουμε το loss ανάμεσα στο output και τις πραγματικές ετικέτες των δειγμάτων του batch. Κάνουμε back propagation στο λάθος με την loss.backward(). Για την ανανέωση των βαρών του νευρωνικού χρησιμοποιούμε την optimizer.step(). Yπολογίζουμε το training loss σε κάθε εποχή, αθροίζοντας τα επιμέρους loss κάθε batch και διαιρώντας με το πλήθος τους και το τυπώνουμε. Σε κάθε εποχή, αφού γίνει η εκπαίδευση μέσω των batches του training set γίνεται αποτίμηση του μοντέλου για κάθε batch του validation set καλώντας την eval_pred και την accuracy_score για τις προβλέψεις των δειγμάτων. Eπιπλέον υπολογίζεται και τυπώνεται το loss στο validation set.
# 
# Ορίζουμε το μοντέλο μας με num_classes = 20 (διαφορετικές ετικέτες), timesteps = 1293, num_features = 140. Χρησιμοποιούμε για loss function το nn.CrossEntropyLoss και ως optimizer το torch.optim.SGD και παραμέτρους epochs = 20, LR = 0.008, weight_decay=0.05. 

# In[9]:


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
            nn.Conv2d(4, 6, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.dropout = nn.Dropout(0.0001)
        self.fc = nn.Linear(int(math.floor(timesteps/16)*math.floor(num_features/16))*8, num_classes)
        
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
        out = self.fc(out)
        return out


# In[10]:


def eval_pred(features):
    output_tensor = model(features.unsqueeze_(1))
    batch_pred = torch.argmax(output_tensor.data, dim=1)
    return batch_pred, output_tensor


# In[11]:


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


# In[12]:


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


# Ακολουθεί το train loop, που τυπώνονται σε κάθε εποχή τα losses σε train και validation set καθώς και τα ποσοστά του μοντέλου στο validation set. Παρατηρούμε ότι το train loss μειώνεται ανά εποχή σταθερά, ενώ το validation loss μειώνεται αλλά πολύ πιο αργά και όχι τόσο σταθερά, γεγονός που μας προϊδεάζει για κακή γενίκευση του μοντέλου. Το ποσοστό στο validation loss αυξάνεται ανά εποχή και φτάνει μέχρι 27% ενώ μετά την 9 εποχή κυμαίνεται γύρω στο 25%.

# In[13]:


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

num_classes = 20
timesteps = 1293
num_features = 140

model = CNN_d2(num_classes,timesteps,num_features)
model.cuda()

print('Training Loop for 2D CNN')

epochs = 30
LR = 0.008
weight_decay=0.05

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay, lr=LR)
earlystopping = EarlyStopping(mode='max', min_delta=0.01, patience=8)
train_val_loop(epochs,model,criterion,optimizer,earlystopping)


# 
# Στη συνέχεια κάνουμε αποτίμηση του 2D CNN στο test set μετά την εκπαίδευση που προηγήθηκε και παρατηρούμε να πιάνει ένα ποσοστό της τάξης του 15%, παρά τα μεγαλύτερα ποσοστά που παρατηρήσαμε στο validation set. Η επιλογή του συγκεκριμένου μοντέλου έγινε μετά από tuning των παραμέτρων LR, weight_decay καθώς και των in, out channels τωv conv2d layers του νευρωνικού. 

# In[14]:


predictions, labels = [], []
for j, data in enumerate(test_loader):
    features_test = torch.tensor(data[0]).float().cuda()
    labels_test = torch.tensor(data[1]).long().cuda()
    
    batch_pred, output_tensor = eval_pred(features_test)

    predictions.append(batch_pred.cpu())
    labels.append(labels_test.cpu())

f1 = accuracy_score(labels, predictions)
print('Score in test set is: %d %%' % (100 * f1))


# In[15]:


print("End of step 9")

