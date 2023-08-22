import datetime

import torch
import os
import pickle
import numpy as np
from model import Spatom
from torch_geometric.data import DataLoader
import torch.optim as optim
import time
import sklearn.metrics as skm
import torch.nn as nn
import matplotlib.pyplot as plt



torch.manual_seed(1209)
np.random.seed(1205)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1209)
    torch.cuda.set_device(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

LEARN_RATE = 0.001
EPOCH = 150
train_set = pickle.load(open('./Data/DBD/data/feature_extract/'+'train'+'_feature.pkl', 'rb'))
test_set = pickle.load(open('./Data/DBD/data/feature_extract/' + 'test' + '_feature.pkl', 'rb'))


def best_f_1(label,output):
    f_1_max = 0
    t_max = 0
    for t in range(1,100):
        threshold = t / 100
        predict = np.where(output>threshold,1,0)
        f_1 = skm.f1_score(label, predict, pos_label=1)
        if f_1 > f_1_max:
            f_1_max = f_1
            t_max = threshold

    pred = np.where(output>t_max,1,0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy,recall,precision,MCC,f_1_max,t_max

def train(train_set=train_set):
    samples_num = len(train_set)
    split_num = int(70/85 * samples_num)
    data_index = np.arange(samples_num)
    np.random.seed(1205)
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    valid_index = data_index[split_num:]
    train_loader = DataLoader(train_set, batch_size=1, sampler=train_index)
    valid_loader = DataLoader(train_set, batch_size=1, sampler=valid_index)
    model = Spatom()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_fun = nn.BCELoss()
    train_log = []
    valid_log = []
    valid_f_1 = []
    max_f_1 = 0
    max_acc = 0
    max_precision = 0
    max_recall = 0
    max_epoch = 0
    max_MCC = 0
    max_t = 0
    for epoch in range(EPOCH):
        start_time = time.time()
        loss = train_epoch(model, train_loader, optimizer, loss_fun)
        train_log.append(loss)
        loss_v,accuracy, recall, precision, MCC, f_1,t_max = valid_epoch(model, valid_loader, loss_fun)
        end_time = time.time()
        valid_log.append(loss_v)
        valid_f_1.append(f_1)
        print("Epoch: ", epoch + 1, "|", "Epoch Time: ", end_time - start_time, "s")
        print("Train loss: ", loss)
        print("valid loss: ", loss_v)
        print('F_1:' ,f_1,t_max)
        print('ACC:' ,accuracy)
        print('Precision: ',precision)
        print('Recall: ', recall)
        print('MCC: ', MCC)
        if f_1 > max_f_1:
            max_f_1 = f_1
            max_acc = accuracy
            max_precision = precision
            max_recall = recall
            max_MCC = MCC
            max_epoch = epoch + 1
            max_t = t_max
            torch.save(model.cpu().state_dict(), f'{result_path}/best_model.dat')
    plt.plot(train_log, 'r-')
    plt.title('train_loss')
    plt.savefig(f"{result_path}/train_loss.png")
    plt.close()

    plt.plot(valid_log, 'r-')
    plt.title('valid_loss')
    plt.savefig(f"{result_path}/valid_log.png")
    plt.close()

    plt.plot(valid_f_1, 'r-')
    plt.title('valid_f_1')
    plt.savefig(f"{result_path}/valid_f_1.png")
    plt.close()

    print("Max Epoch: ", max_epoch)
    print('F_1:', max_f_1, max_t)
    print('ACC:', max_acc)
    print('Precision: ', max_precision)
    print('Recall: ', max_recall)
    print('MCC: ', max_MCC)

def train_epoch(model,train_loader,optimizer,loss_fun):
    model.to(DEVICE)
    model.train()
    loss = 0
    num = 0
    for step, data in enumerate(train_loader):
        feature = torch.autograd.Variable(data.x.to(DEVICE,dtype=torch.float))
        label = torch.autograd.Variable(data.y.to(DEVICE, dtype=torch.float))
        dist = torch.autograd.Variable(data.dist.to(DEVICE, dtype=torch.float))
        adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))

        optimizer.zero_grad()
        pred = model(feature,dist,adj)
        train_loss = loss_fun(pred, label)
        train_loss.backward()
        optimizer.step()
        loss = loss + train_loss.item()
        num += 1
    epoch_loss = loss / num
    return epoch_loss

def valid_epoch(model,valid_loader,loss_fun):
    model.to(DEVICE)
    model.eval()
    loss = 0
    num = 0
    all_label = []
    all_pred  = []
    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            feature = torch.autograd.Variable(data.x.to(DEVICE, dtype=torch.float))
            label = torch.autograd.Variable(data.y.to(DEVICE, dtype=torch.float))
            dist = torch.autograd.Variable(data.dist.to(DEVICE, dtype=torch.float))
            adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))

            pred = model(feature,dist,adj)
            valid_loss = loss_fun(pred, label)
            pred = pred.cpu().numpy()
            all_label.extend(label.cpu().numpy())
            all_pred.extend(pred)
            loss = loss + valid_loss.item()
            num += 1
    epoch_loss = loss / num
    accuracy, recall, precision, MCC, f_1_max, t_max = best_f_1(np.array(all_label),np.array(all_pred))
    return epoch_loss,accuracy, recall, precision, MCC, f_1_max,t_max


def every_best_f_1(label, output):
    # f_1_max = 0
    t_max = 0.4
    # for t in range(1,100):
    #     threshold = t / 100
    #     predict = np.where(output>threshold,1,0)
    #     f_1 = skm.f1_score(label, predict, pos_label=1)
    #     if f_1 > f_1_max:
    #         f_1_max = f_1
    #         t_max = threshold

    pred = np.where(output > t_max, 1, 0)
    f_1_max = skm.f1_score(label, pred, pos_label=1)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy, recall, precision, MCC, f_1_max, t_max


def test(test_set):
    test_loader = DataLoader(test_set, batch_size=1)
    model = Spatom().to(DEVICE)
    model.load_state_dict(torch.load(f'{result_path}/best_model.dat'))
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
            # accuracy, recall, precision, MCC, f_1, t_max = every_best_f_1(np.array(label), np.array(predict_protein))
            # every_F1.append(f_1)
            # label = np.array(label)
            # predict_protein = np.array(predict_protein)
            # positive = predict_protein[np.where(label>0)[0]]
            # negtive = predict_protein[np.where(label<1)[0]]
            # pos_median = np.median(positive)
            # neg_median = np.median(negtive)
            # median_every = pos_median - neg_median
            # median.append(median_every)
        # print(every_F1)
        # print(median)
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
    path_dir = "./result"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    result_path = f'{path_dir}/{localtime}'
    # result_path = './result/2022-12-05-18:47:00'
    os.makedirs(result_path)
    train()
    test(test_set)

