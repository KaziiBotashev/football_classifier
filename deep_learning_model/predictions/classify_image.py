from deep_learning_model.training.models import Classifier
from deep_learning_model.training.config import CLASSES_ALL, CLASSES_1, CLASSES_2
from deep_learning_model.training.img_transformations import SquarePad

import torch
import torchvision.transforms as transforms

import os

class ImageClassifier:
    def __init__(self): 
        models_dir_path = 'deep_learning_model/trained_model/soccernet-'
         
        self.classifier_0 = Classifier(0).load_from_checkpoint(models_dir_path + '0' +'.ckpt', team_num = 0)
        self.classifier_0.eval()
        self.classifier_0.freeze()

        self.classifier_1 = Classifier(1).load_from_checkpoint(models_dir_path + '1' +'.ckpt', team_num = 1)
        self.classifier_1.eval()
        self.classifier_1.freeze()

        self.classifier_2 = Classifier(2).load_from_checkpoint(models_dir_path + '2' +'.ckpt', team_num = 2)
        self.classifier_2.eval()
        self.classifier_2.freeze()

        self.classifier_3 = Classifier(3).load_from_checkpoint(models_dir_path + '3' +'.ckpt', team_num = 3)
        self.classifier_3.eval()
        self.classifier_3.freeze()

    def predict(self, image, use_individual_models):
                    
        transforms_image = transforms.Compose([SquarePad(), transforms.Resize((224,224)),
                                               transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])
        image = transforms_image(image) 
        image = image.unsqueeze(0)
        
        if use_individual_models:
            team_num = torch.argmax(torch.exp(self.classifier_0(image)), dim = 1)
            print(team_num)
            if team_num == 0:
                player_id = torch.argmax(torch.exp(self.classifier_1(image)), dim = 1)
                print(player_id)
                return CLASSES_1[player_id]
            elif team_num == 1:
                player_id = torch.argmax(torch.exp(self.classifier_2(image)), dim = 1)
                print(player_id)
                return CLASSES_2[player_id]
            elif team_num == 2:
                return "C-"
            elif team_num == 3:
                return "D-"
            elif team_num == 4:
                return "other"
        else:
            output = self.classifier_3(image) 
            output = torch.exp(output)   
            class_idx = torch.argmax(output, dim=1)    
            
            return CLASSES_ALL[class_idx]
