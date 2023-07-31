from PIL import Image
from torchvision import datasets
from torch.utils.data.dataset import Dataset

import random
import numpy as np
import pandas as pd
import PIL.ImageOps
import pathlib
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from parser_define import train_args_define



# Dataset class for microCT dataset 
class Microct_large_Dataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True, train=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.train = train
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        df_rock = pd.read_csv(self.imageFolderDataset.root + '/../microct_0114_random_labels.csv') 
        df_rock['Permeability'] = df_rock['Permeability'] / (df_rock['Pixelsize'] * df_rock['Pixelsize']) # Voxel normalized permeability
        df_rock = df_rock.drop(columns=['Label', 'GI', 'NI', 'Size', 'Depth', 'Image', 'Pixelsize'])

        # Permeability scaling
        df_rock['Permeability'] = np.log10(df_rock['Permeability']) / 5 # log10

        # Make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        # Training set setting, we randomized images and save into folders such as AA01, AA02, ..
        if self.train == True:
            img_path = pathlib.PurePath(img0_tuple[0])
            #print("parent name", img_path.parent.name)
            
            random_label_train = img_path.parent.name
            data0 = df_rock.loc[df_rock['Created_label'] == random_label_train]
            data1 = df_rock.loc[df_rock['Created_label'] == random_label_train]
            data0 = torch.tensor(data0.drop(columns=['Created_label']).values).float()
            data1 = torch.tensor(data1.drop(columns=['Created_label']).values).float() 

        # Validation set setting
        else:
            img_path = pathlib.PurePath(img0_tuple[0])
            #print("parent name", img_path.parent.name)
            random_label_test = img_path.parent.name
            data0 = df_rock.loc[df_rock['Created_label'] == random_label_test]
            data1 = df_rock.loc[df_rock['Created_label'] == random_label_test]
            data0 = torch.tensor(data0.drop(columns=['Created_label']).values).float()
            data1 = torch.tensor(data1.drop(columns=['Created_label']).values).float()   

        # If the difference is less than 0.1, it's considered as similar 
        if (torch.sum(torch.abs((data0-data1)/data1)) < 0.1):
            label_rock = torch.tensor(1).float()
        else:
            label_rock = torch.tensor(0).float()

        return img0, img1, data0, data1, torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32)), label_rock
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# for test
if __name__ == '__main__':
    train_args = train_args_define()
    train_args.datapath = os.getcwd() + '/data/microct_split'
    data_set = Microct_large_Dataset(datasets.ImageFolder(root=train_args.datapath))
    print(data_set)