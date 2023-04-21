import torch
import time

import model
import loader
import utils

import aug

# Setting the path to the directory containing the images
images_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/Reddit Skin Lesions/images"

# Setting the percentage of images to use for training (80%)
train_percent = 0.8

# defining the path to the JSON file containing information about skin lesion images
info_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/Reddit Skin Lesions/dataset.json"

# getting the data information in csv format
csv = utils.get_data_info(info_path)

# create a dictionary that maps subreddits to numeric labels
label_values = {'Dermatology': 0, 'skincancer': 1, 'skin': 2, 'Skincare_Addiction': 3, 'Rosacea': 4, 'skin': 5, 'Psoriasis': 6,
                'SkincareAddiction': 7, '30PlusSkinCare': 8, 'Warts': 9, 'peeling': 10, 'Acne': 11, 'skincancer': 12, 'eczema': 13,
                'popping': 14, 'SkincareAddicts': 15}

# Getting image paths and corresponding labels
train_files, test_files = utils.get_image_groups(images_path, train_percent)

train_labels = utils.get_labels(train_files, label_values, csv)
test_labels = utils.get_labels(test_files, label_values, csv)

# Generating the datasets and dataloaders
train_dataset = loader.MyDataset(train_files, train_labels, my_transform=None)
test_dataset = loader.MyDataset(test_files, test_labels, my_transform=None)
batch_size = 30
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Setting configs
model = model.MyModel(num_classes=16)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Start training
start_time = time.time()
for epoch in range(num_epochs):
    for k, (batch_images, batch_labels, id_img) in enumerate(train_dataloader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = model(batch_images)
        loss = loss_func(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

    print (f"- Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.4f}")         

# making inferences
with torch.no_grad():
    correct, total = 0, 0
    for images, labels, img_id in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

finish_time = time.time()
print(f"Executado em {(finish_time - start_time)/60} minutos")