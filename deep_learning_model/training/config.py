import torchvision.transforms as transforms
from torchvision.models import resnet34

try:
    from model_architecture import SoccerNet
    from img_transformations import SquarePad
except:
    from deep_learning_model.training.model_architecture import SoccerNet
    from deep_learning_model.training.img_transformations import SquarePad
    

TRANSFORM = transforms.Compose(
    [transforms.RandomRotation((-10,10)), SquarePad(), transforms.Resize((224,224)),
     transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], 
     std=[0.229, 0.224, 0.225])])

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
MODEL = SoccerNet() ## change it to True before training
NUM_CLASSES = 25
LEARNING_RATE = 10e-3
#CLASSES = ('B5', 'B48', 'B4', 'B1', 'A11', 'B25', 'B22', #numerical order
#           'A8', 'C-', 'B34', 'B10', 'B8', 'A98', 'A55', 
#           'B27', 'A15', 'A19', 'A20', 'A33', 'A5', 'D-',
#           'A57', 'B44', 'A31', 'other')

CLASSES = ('B5', 'B48', 'B10', 'B8', 'A98', 'A55', 'B27', 'A15', #alphabetical order due to ImageFolder 
           'A19', 'A20', 'A33', 'A5', 'B4', 'D-', 'A57',
           'B44', 'A31', 'other', 'B1', 'A11', 'B25', 'B22',
           'A8',  'C-', 'B34')


MODEL_NAME = 'soccerenet-epoch=37-val_loss=0.16.ckpt'
