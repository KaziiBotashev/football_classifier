from models import Classifier
from img_transformations import SquarePad

import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl

import os

import numpy as np

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir_path = '../trained_model/soccernet-'
    team1 = [0, 1, 2, 3, 5, 6, 9, 10, 11, 14, 22]
    team2 = [4, 7, 12, 13, 15, 16, 17, 18, 19, 21, 23]

    model_0 = Classifier(0).load_from_checkpoint(
        models_dir_path + '0' + '.ckpt', team_num=0).to(device)
    model_0.eval()
    model_0.freeze()

    model_1 = Classifier(1).load_from_checkpoint(
        models_dir_path + '1' + '.ckpt', team_num=1).to(device)
    model_1.eval()
    model_1.freeze()

    model_2 = Classifier(2).load_from_checkpoint(
        models_dir_path + '2' + '.ckpt', team_num=2).to(device)
    model_2.eval()
    model_2.freeze()

    model_3 = Classifier(3).load_from_checkpoint(
        models_dir_path + '3' + '.ckpt', team_num=3).to(device)
    model_3.eval()
    model_3.freeze()

    pl.seed_everything(42)
    model_3.prepare_data()
    dataloader = model_3.test_dataloader()

    indices_alphabetical = [
        0,
        1,
        12,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17]

    preds = []
    prob_preds = []
    labels = []
    k = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model_0(x)
        preds_0 = torch.exp(y_hat)
        preds_0 = torch.argmax(preds_0, dim=1)
        predicted_labels = []
        for i, pred_0 in enumerate(preds_0):
            #            print(x[i].shape)
            if pred_0 == 0:
                player_id = torch.argmax(
                    torch.exp(
                        model_1(
                            x[i].unsqueeze(0))),
                    dim=1)
#                print(player_id)
                predicted_labels.append(indices_alphabetical[team1[player_id]])
            elif pred_0 == 1:
                player_id = torch.argmax(
                    torch.exp(
                        model_2(
                            x[i].unsqueeze(0))),
                    dim=1)
#                print(player_id)
                predicted_labels.append(indices_alphabetical[team2[player_id]])
            elif pred_0 == 2:
                predicted_labels.append(23)
            elif pred_0 == 3:
                predicted_labels.append(13)
            elif pred_0 == 4:
                predicted_labels.append(17)
        preds += predicted_labels
        labels += (y.cpu().numpy().reshape(-1).tolist())
        print(k)
        k += 1
    preds = np.array(preds).reshape(-1)
    labels = np.array(labels).reshape(-1)
    print(preds)
    print(labels)

    balanced_acc = balanced_accuracy_score(preds, labels)
    print("Accuracy: ", balanced_acc)
    print(classification_report(preds, labels))
