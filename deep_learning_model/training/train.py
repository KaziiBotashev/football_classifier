import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import Classifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys
try:
    import config
except:
    from deep_learning_model.training import config

team_num = int(sys.argv[1])
print("Team number: ", team_num)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='../trained_model',
    filename='soccernet-'+str(team_num),
    save_top_k=1,
    mode='min',
    save_weights_only = True
)

print("Team number: ", team_num)



pl.seed_everything(42)

# logger = TensorBoardLogger('tb_logs', name='SoccerNet')

model = Classifier(team_num)
trainer = pl.Trainer(max_epochs=200, gpus=1, progress_bar_refresh_rate = 25, callbacks = [checkpoint_callback, EarlyStopping(monitor='val_loss_epoch',patience = 15)])

# start training
trainer.fit(model)
# save trained model
# torch.save(model.state_dict(), config.MODEL_NAME)
# test on test data
trainer.test(model)


