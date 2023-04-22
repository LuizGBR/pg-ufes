import torch
from torchvision import models
import pytorch_lightning


class MyModel(pytorch_lightning.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Remove the last layer (the fully-connected layer) from the ResNet model
        self.features =torch.nn.Sequential(*list(resnet.children())[:-1])
        # Add a linear classifier on top of the ResNet features
        self.classifier = torch.nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Pass the input tensor through the ResNet features
        x = self.features(x)
        # Flatten the features into a 1D tensor
        x = x.view(x.size(0), -1)
        # Pass the features through the linear classifier
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        batch_images, batch_labels = batch    
        outputs = self(batch_images)
        loss = torch.nn.CrossEntropyLoss()(outputs, batch_labels)
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

