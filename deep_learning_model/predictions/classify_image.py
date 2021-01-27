from deep_learning_model.training.models import Classifier
from deep_learning_model.training.config import CLASSES, MODEL_NAME
from deep_learning_model.training.img_transformations import SquarePad

import torch
import torchvision.transforms as transforms

import os

class ImageClassifier:
    def __init__(self):        
        self.classifier = Classifier()
        model_path = os.path.join('deep_learning_model', 'trained_model', MODEL_NAME)   
                      
#        self.classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.classifier.load_from_checkpoint(model_path)
        self.classifier.eval()

    def predict(self, image):
                    
        transforms_image = transforms.Compose([transforms.RandomRotation((-10,10)), SquarePad(), transforms.Resize((224,224)),
                                               transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])
        image = transforms_image(image) 
        image = image.unsqueeze(0)
        output = self.classifier(image) 
        output = torch.exp(output)   
        class_idx = torch.argmax(output, dim=1)    
        
        return CLASSES[class_idx]
