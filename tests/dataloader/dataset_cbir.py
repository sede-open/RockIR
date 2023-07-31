import glob
import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

current_path = os.getcwd()

from torchvision import transforms


class Microct_Dataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(image_paths+'/*.png') # image_paths
        self.transform = transform
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        t_image = image.convert("RGB") # convert from greyscale to RGB 
        idx_db_test = self.image_paths[index].index("db_test")
        label = self.image_paths[index][idx_db_test+8:idx_db_test+10]
        # print (index, self.image_paths[index])

        if self.transform is not None:
            t_image = self.transform(t_image) # into tensor       
        
        return t_image, label
    
    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    microct_dataset = Microct_Dataset(image_paths='/home/simuser/poreimage_cbir/images/microct', 
                                            transform=transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                                ]))

    test_loader = torch.utils.data.DataLoader(
        microct_dataset,
        batch_size=1, shuffle=False)
