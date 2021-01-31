from model_architecture import SoccerNet
from models import Classifier
import sys
import config
import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle


if __name__ == '__main__':
    team_num = int(sys.argv[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Team number: ", team_num)
    if team_num == 0:
        num_classes = 5
    elif team_num == 1:
        num_classes = 11
    elif team_num == 2:
        num_classes = 11
    elif team_num == 3:
        num_classes = 25
    model_name = "soccernet-" + str(team_num) + '.ckpt'
    model = Classifier(team_num).load_from_checkpoint(
        "../trained_model/" + model_name, team_num=team_num)
    model.eval()
    model.freeze()
    model = model.to(device)
    pl.seed_everything(42)
    model.prepare_data()
    dataloader = model.test_dataloader()

    preds = []
    prob_preds = []
    labels = []
    k = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        pred = torch.exp(y_hat)
        prob_preds.append(pred.cpu().numpy())
        pred = torch.argmax(pred, dim=1).cpu().numpy().reshape(-1).tolist()
        preds += (pred)
        labels += (y.cpu().numpy().reshape(-1).tolist())
        print(k)
        k += 1
    prob_preds = np.array(prob_preds)
    preds = np.array(preds).reshape(-1)
    labels = np.array(labels).reshape(-1)

    balanced_acc = balanced_accuracy_score(preds, labels)
    print("Accuracy: ", balanced_acc)
    print(classification_report(preds, labels))

    y_score = np.vstack(prob_preds.tolist())
    y_test = label_binarize(labels, classes=range(num_classes))

    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(30, 15))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    cm = plt.get_cmap('gist_rainbow')
    colors = cycle([cm(1. * i / num_classes) for i in range(num_classes)])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.025, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_" + model_name + "_" + str(balanced_acc) + ".png")
    plt.show()
