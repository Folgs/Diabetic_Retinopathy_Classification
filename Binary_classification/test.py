from cv2 import dft
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import pandas as pd
from PIL import Image

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision import transforms
torch.set_grad_enabled(True)

import shutil
import copy
import os


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import datetime
from time import time

import wandb

import seaborn as sn
import itertools
import cv2

print("\n") 

if(torch.cuda.is_available()):
    print(" > > > GPU RUNNING \n")
else:
    print(" > > > GPU NOT AVAILABLE \n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_dir = "C:/Users/itoxi/Documents/Projects/DLAI/Project/Binary classification/"
path_csv = "C:/Users/itoxi/Documents/Projects/DLAI/Project/Binary classification/loaded_data.csv"


def weighted_binary_cross_entropy(output, target, weights= None):
    output = torch.clamp(output, min=1e-8, max = 1-1e-8)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    
    return torch.neg(torch.mean(loss))
        
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (300, 300))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

def preprocess_images(dir_path, csv_path):
    # Preprocessing of images. Transformations.
    # Do it only once.

    n_zeros = 0
    n_ones = 0

    first_run = False
    num_run = 102

    image_data = pd.read_csv(csv_path, sep = ';', encoding="latin-1")
    N_images = image_data.shape[0]
    transform = lambda y: transforms.Compose([ transforms.CenterCrop(y), transforms.Resize([300,300])]) 
    for i in range(0,N_images):
        ID_image, multilabel = image_data.loc[i,['ID_Imatge','Retinopatia']]
        path = f"{dir_path}/{multilabel}/{ID_image}"
        try:
            image = Image.open(path)
            W, H = image.size
            image = transform(H)(image)
            image = np.array(image)
            image = load_ben_color(image)
            image = Image.fromarray(image)
            if int(multilabel) == 1:
                label = '0'
                n_zeros = n_zeros + 1
            else:
                label = '1'
                n_ones = n_ones + 1
            image.save(f"{dir_path}/classe_{label}/{ID_image}")
            print(f"{dir_path}/classe_{label}/{ID_image}")
        except:
            print(f"could not open image/classe_{multilabel}/{ID_image}")  

    print ("-----------------------------\n")
    print ("Number of 0's = " + str(n_zeros))
    print ("Number of 1's = " + str(n_ones) + "\n")
    print ("-----------------------------\n")
    
    return

def create_csv_from_loaded_images(dir_path, dir_csv): 
    # Create a CSV of the loaded images and their label. DO ONLY ONCE:

    file_names = []
    file_labels = []

    # load all "1" images
    for filename in os.listdir(dir_path + "classe_1/"):
        name = os.path.splitext(filename)[0]
        file_names.append(name)
        file_labels.append(1)
    df1 = pd.DataFrame(list(zip(file_names,file_labels)), columns=["image_id", "labels"])

    df1.head()

    file_names = []
    file_labels = []

    # load all "0" images
    for filename in os.listdir(dir_path + "classe_0/"):
        name = os.path.splitext(filename)[0]
        file_names.append(name)
        file_labels.append(0)
    df0 = pd.DataFrame(list(zip(file_names,file_labels)), columns=["image_id", "labels"])

    df0.head()
    
    # merge both dataframes
    df = pd.concat([df0, df1], ignore_index=True)

    # save csv to drive folder
    df.to_csv(dir_csv)  
    print ( " > > > Images tracked and ID's + labels loaded in .csv file! \n")
    return

class Diabetic_Retinopathy_Dataset(Dataset): 

  def __init__(self):
    self.image_data = pd.read_csv(path_csv, sep=',', encoding="latin-1")
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(256), transforms.Normalize(0,1)]) 
    self.image0 = None
    self.image1 = None
    self.buffer0 = False
    self.buffer1 = False

  def __getitem__(self, i):
    ID_imatge, label = self.image_data.loc[i,['image_id','labels']]
    label = int(label)
    #print (f"{path_dir}classe_{label}/{ID_imatge}.jpg")
    try:
      path = f"{path_dir}classe_{label}/{ID_imatge}.jpg"
      image = Image.open(path)
      image = self.transform(image)
      if ((label==0) & (self.buffer0 == False)):
        self.image0 = image
        self.buffer0 ==True
      if ((label==1) & (self.buffer1 == False)):
        self.image1 = image
        self.buffer1 = True
        
    except:
      
      if label == 1 & self.buffer1 == True:
        image = self.image1
      elif label == 1 & self.buffer1 == False:
        print(f"could not open image/classe_{label}/{ID_imatge}")     
      
      if label == 0 & self.buffer0 == True:
        image = self.image0
      elif label == 0 & self.buffer0 == False:
        image = self.image1
        print(f"could not open image/classe_{label}/{ID_imatge}")

    return image, int(label)

  def __len__(self):
    return len(self.image_data)

def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = original_lr * (0.1 ** (epoch // 30))
    # For some models, different parameters are in different groups with different lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def correct_predictions(predicted_batch, label_batch):
  print (predicted_batch)
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()

  return acum

def get_accuracy(predicted_batch, label_batch):
    #print (predicted_batch)
    pred = predicted_batch.round()
    #print (pred)
    return pred.eq(label_batch.view_as(pred)).sum().item()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    
    # save an extra copy if it is the best model yet
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')  

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig
    
def train(train_loader, model, criterion, optimizer, device):

    # Switch to train mode
    model.train()
    loss_mean = []
    acc_mean = []

    for i, (images, target) in enumerate(train_loader):
        # Reset gradients
        optimizer.zero_grad()

        #move images to gpu
        images = images.to(device)
        target = target.to(device)

        target = target.unsqueeze(1).float()

        # Compute output
        output = model(images)

        #loss = criterion(output,target)
        loss = weighted_binary_cross_entropy(output, target, torch.tensor([N_images/N_zeros, N_images/N_ones]))
        acc = 100 * (get_accuracy(output, target) / images.shape[0])

        loss.backward()

        loss_mean.append(loss.item())
        acc_mean.append(acc)

        optimizer.step()

    return np.mean(acc_mean), np.mean(loss_mean)

def validate(val_loader, model, criterion, device):

    # Switch to evaluate mode
    model.eval()

    # We will save the values of the accuracies in this list to return the mean of the whole dataset at the end
    acc = 0
    loss = []

    y_pred = []
    y_true = []

    with torch.no_grad():  # We do not need to compute gradients
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)
            target = target.unsqueeze(1).float()

            output = model(images)

            # loss
            #loss.append(criterion(output, target).item())
            loss.append(weighted_binary_cross_entropy(output, target, torch.tensor([N_images/N_zeros, N_images/N_ones])).item())
            # measure accuracy
            acc += 100 * get_accuracy(output, target)

            # confusion matrix  
            #print(output)         
            output = output.round().data.cpu().numpy()
            
            y_pred.extend(output) # Save Prediction
            
            target = target.data.cpu().numpy()
            y_true.extend(target) # Save Truth

    cf_matrix = confusion_matrix(y_true, y_pred)
    
    eval_acc = acc / len(val_loader.dataset)
    
    print (eval_acc)
    print(np.mean(loss))
    print (cf_matrix)

    return eval_acc, np.mean(loss), cf_matrix

def get_run_number(dir_path):
    try:
        with open( dir_path + 'run_num.txt', 'r' ) as fle:
            counter = int( fle.readline() ) + 1
    except FileNotFoundError:
        counter = 0

    with open( dir_path + 'run_num.txt', 'w' ) as fle:
        fle.write( str(counter) )
    return counter


preprocess_data = False
create_csv = False
run_training= True

if preprocess_data:
    preprocess_images(path_dir, path_csv)

if create_csv:
    create_csv_from_loaded_images(path_dir, path_csv)


    
df = pd.read_csv(path_csv, sep = ',', encoding="latin-1")
N_images = df.shape[0]

N_zeros = len(os.listdir(path_dir + "classe_0/"))
N_ones = len(os.listdir(path_dir + "classe_1/"))

print (' > > > we have ' + str(N_images) + ' images')
print (' \t  # of 0 : ' + str(N_zeros) + ' images')
print (' \t  # of 1 : ' + str(N_ones) + ' images \n')



if run_training:
    torch.backends.cudnn.benchmark = True

    epochs = 50
    learning_rate= 1e-6
    batch_size = 256
    num_run = get_run_number(path_dir)

    classes = ('0: Normal','1: Patologia')

    dataset__training, dataset_validation = torch.utils.data.random_split(Diabetic_Retinopathy_Dataset(), [round(N_images*0.8),round(N_images*0.2)]) 
    dataloader_training = DataLoader(dataset__training, batch_size=batch_size, shuffle=True, pin_memory= True)
    dataloader_validating = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False, pin_memory= True)

    x, labels = next(iter(dataloader_training))

    print(' > > > shape of data: ')
    print('\t Databunch of images' + str(x.shape))
    print('\t Databunch of labels' + str(labels.shape))

    # Take the ResNet50 model
    model = models.resnet34(pretrained = True).to(device)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs,256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256,1),
        nn.Sigmoid()
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # We use wandb to log our results
    wandb.login()

    try:
        wandb.finish() # This is needed just in case there was a wandb run from a previous execution
    except:
        pass

    wandb.init(project="retino_binary")
    wandb.run.name = f'run_{num_run}'
    epochs = 50
    best_acc = 0.
    
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)

        acc_tr, loss_tr = train(dataloader_training, model, criterion, optimizer, device)

        wandb.log({'Training/accuracy':acc_tr}, epoch)
        wandb.log({'Training/loss':loss_tr}, epoch)

        acc, loss, cf_matrix = validate(dataloader_validating, model, criterion,device)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        wandb.log({'Validation/accuracy': acc},epoch)
        wandb.log({'Validation/loss':loss},epoch)
        wandb.log({'Conf_matrix':plot_confusion_matrix(cf_matrix, classes,normalize=False)}, epoch) 

        print("Iteracio", epoch,"completada. Validate loss", loss, "train loss", loss_tr, "Validate acc", acc, "train acc", acc_tr)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "resnet18",
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


    wandb.finish()







else:
    print ("not running training")

