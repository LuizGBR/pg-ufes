import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import torch
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Definig the Dataset class
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_path, labels, my_transform = None):
        super().__init__()
        
        self.imgs_path = imgs_path
        self.labels = labels

        if my_transform is not None:
            self.transform = my_transform
        else:
            self.transform = transforms.ToTensor()
    
    def __len__(self):

        return len(self.imgs_path)
    
    def __getitem__(self, item):
        img = Image.open(self.imgs_path[item]).convert("RGB")
        
        image = self.transform(img) # Applying the transformations

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels