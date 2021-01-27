import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

try:
    import config
except:
    from deep_learning_model.training import config

class Classifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.transform = config.TRANSFORM
        self.train_batch_size = config.TRAIN_BATCH_SIZE
        self.val_batch_size = config.VAL_BATCH_SIZE
        self.test_batch_size = config.TEST_BATCH_SIZE        
        self.model = config.MODEL
        self.learning_rate = config.LEARNING_RATE  

    def forward(self, x):
        x = self.model(x)
        return x    

    def cross_entropy_loss(self, outputs, labels):        
        return nn.CrossEntropyLoss()(outputs, labels)

    def prepare_data(self):
        self.trainset = datasets.ImageFolder("../data/images_splited_balanced_upscaled/train", transform=self.transform)
        self.valset = datasets.ImageFolder("../data/images_splited_balanced_upscaled/val", transform=self.transform)
        self.testset = datasets.ImageFolder("../data/images_splited_balanced_upscaled/test", transform=self.transform)
        


    def train_dataloader(self):        
        trainloader = DataLoader(self.trainset, batch_size=self.train_batch_size,
                                          shuffle=True)
        return trainloader                                             

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.val_batch_size)

    def test_dataloader(self):        
        testloader = DataLoader(self.testset, batch_size=self.test_batch_size,
                                         shuffle=False)                                         
        return testloader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss(y_hat, y)

        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Val",
                                            avg_loss,
                                            self.current_epoch)
        print("Loss/Val", avg_loss, self.current_epoch)
        epoch_dictionary={
            # required
            'val_loss_epoch': avg_loss}
        self.log('val_loss_epoch', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': self.cross_entropy_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        
    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)
        print("Loss/Train", avg_loss, self.current_epoch)
        epoch_dictionary={
            'loss_epoch': avg_loss}
        self.log('loss_epoch', avg_loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
       'optimizer': optimizer,
       'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose = True, patience = 5),
       'monitor': 'val_loss_epoch'
        }
