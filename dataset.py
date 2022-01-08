from PIL import Image
import torch

from io import BytesIO
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self , 
                transform , 
                path = 'celeb_dataset', 
                resolution = 256):
        super(Dataset , self).__init__()
        print(path)
        self.files = os.listdir(path)
        self.path = path
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self , idx):
        img = self.files[idx]
        img = Image.open(os.path.join(self.path , img))
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    dataset = Dataset(None)
