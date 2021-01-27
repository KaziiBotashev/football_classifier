import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import Classifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
try:
    import config
except:
    from deep_learning_model.training import config

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='models',
    filename='soccerenet-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)




pl.seed_everything(42)

# logger = TensorBoardLogger('tb_logs', name='SoccerNet')
model = Classifier()
trainer = pl.Trainer(max_epochs=200, gpus=1, progress_bar_refresh_rate = 25, callbacks = [checkpoint_callback, EarlyStopping(monitor='val_loss_epoch',patience = 9)])

# start training
trainer.fit(model)
# save trained model
torch.save(model.state_dict(), config.MODEL_NAME)
# test on test data
trainer.test(model)
