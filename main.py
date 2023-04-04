import torch
import pandas
import time

import model
import loader

# Setting the path to the directory containing the images
images_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/HAM10000/images"

# Setting the percentage of images to use for training (80%)
train_percent = 0.8

# defining the path to the CSV file containing information about skin lesion images
info_path = "/Users/LG - Workspace/Documents/Machine Learning/Datasets/HAM10000/HAM10000_metadata"

# reading the CSV file into a Pandas DataFrame object, using the "image_id" column as the index column
csv = pandas.read_csv(info_path, index_col="image_id").squeeze("columns")

# create a dictionary that maps skin lesion types to numeric labels
label_values = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df':3 , 'mel': 4, 'nv': 5, 'vasc': 6 }

# Getting image paths and corresponding labels
train_files, test_files = loader.get_image_batches(images_path, train_percent)

train_labels = loader.get_labels(train_files, label_values, csv)
test_labels = loader.get_labels(test_files, label_values, csv)

# Generating the datasets and dataloaders
train_dataset = loader.MyDataset(train_files, train_labels, my_transform=None)
test_dataset = loader.MyDataset(test_files, test_labels, my_transform=None)
batch_size = 30
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Setting configs
model = model.MyModel(num_classes=7)
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