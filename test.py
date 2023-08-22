import torch
import pickle
import numpy as np
from model import Spatom
from torch_geometric.data.dataloader import DataLoader
import sklearn.metrics as skm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

test_set = pickle.load(open('./Data/DBD/data/feature_extract/' + 'test' + '_feature.pkl', 'rb'))


def best_f_1(label, output):
    f_1_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100
        predict = np.where(output > threshold, 1, 0)
        f_1 = skm.f1_score(label, predict, pos_label=1)
        if f_1 > f_1_max:
            f_1_max = f_1
            t_max = threshold

    pred = np.where(output > t_max, 1, 0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy, recall, precision, MCC, f_1_max, t_max


def test(test_set=test_set):
    test_loader = DataLoader(test_set, batch_size=1)
    model = Spatom().to(DEVICE)
    model.load_state_dict(torch.load('./result/model/best_model.dat'))
    model.eval()
    all_label = []
    all_pred = []
    # every_F1 = []
    median = []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            feature = torch.autograd.Variable(data.x.to(DEVICE, dtype=torch.float))
            label = torch.autograd.Variable(data.y.to(DEVICE, dtype=torch.float))
            dist = torch.autograd.Variable(data.dist.to(DEVICE, dtype=torch.float))
            adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))
            pos = data.POS[0]
            length = data.length.item()

            pred = model(feature, dist, adj)
            pred = pred.cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            predict_protein = [0] * length
            for k, i in enumerate(pos):
                predict_protein[i] = pred[k]
            all_label.extend(label)
            all_pred.extend(predict_protein)

    accuracy, recall, precision, MCC, f_1, t_max = best_f_1(np.array(all_label), np.array(all_pred))
    AUC = skm.roc_auc_score(all_label, all_pred)
    precisions, recalls, thresholds = skm.precision_recall_curve(all_label, all_pred)
    AUPRC = skm.auc(recalls, precisions)
    print("test: ")
    print('F_1:', f_1, t_max)
    print('ACC:', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('MCC: ', MCC)
    print('AUROC: ', AUC)
    print('AUPRC: ', AUPRC)


if __name__ == '__main__':
    test()
