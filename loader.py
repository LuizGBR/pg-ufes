import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import torch
import random
import os
from PIL import Image

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

        resized_image = fn.resize(img, size=[224,224]) # Resizing to resnet default size
        
        image = self.transform(resized_image) # Applying the transformations

        img_id = self.imgs_path[item].split('\\')[-1].split('.')[0]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, img_id

# Defining util functions

def get_image_batches(path, train_percent):

    # Get a list of all the image file names in the directory
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

    # Calculate the number of images to use for training and testing
    num_train = int(len(image_files) * train_percent)
    num_test = len(image_files) - num_train

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Split the shuffled list of image files into two separate lists
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]

    return train_files, test_files

def get_labels(images, label_values, csv):
    labels = list()

    for path in images:
        key = path.split('\\')[-1].split('.')[0]
        labels.append(label_values[csv.dx[key]])
    
    return labels