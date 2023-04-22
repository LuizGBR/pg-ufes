import pytorch_lightning
import torch
from torchvision import models
from torchvision import transforms

import model
import loader
import utils

# Setting the path to the directory containing the images
images_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/Reddit Skin Lesions/images"

# defining the path to the JSON file containing information about skin lesion images
info_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/Reddit Skin Lesions/dataset.json"

# getting the data information in csv format
csv = utils.get_data_info(info_path)

# create a dictionary that maps subreddits to numeric labels
label_values = {'Dermatology': 0, 'skincancer': 1, 'skin': 2, 'Skincare_Addiction': 3, 'Rosacea': 4, 'skin': 5, 'Psoriasis': 6,
                'SkincareAddiction': 7, '30PlusSkinCare': 8, 'Warts': 9, 'peeling': 10, 'Acne': 11, 'eczema': 12,
                'popping': 13, 'SkincareAddicts': 14}

# Setting the percentage of images to use for training (80%)
train_percent = 0.8

# Getting image paths and corresponding labels
train_files, test_files = utils.get_image_groups(images_path, train_percent)

train_labels = utils.get_labels(train_files, label_values, csv)
test_labels = utils.get_labels(test_files, label_values, csv)

# Generating the datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = loader.MyDataset(train_files, train_labels, my_transform=transform)
test_dataset = loader.MyDataset(test_files, test_labels, my_transform=transform)
batch_size = 10
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#setting and running the pytorch lightning pipeline

model = model.MyModel(num_classes=len(label_values)+1)

trainer = pytorch_lightning.Trainer(limit_train_batches=30, max_epochs=10)
trainer.fit(model=model, train_dataloaders=train_dataloader)