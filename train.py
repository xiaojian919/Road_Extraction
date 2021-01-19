import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim
from dataset import MyDataset
from Models import *
from loss import *


class Trainer:
    def __init__(self, net, loss_func, save_path, log, traindataset_path, valdataset_path, batchsize=16):
        '''
        :param net: 选择的网络
        :param loss_func: 损失函数
        :param save_path: 参数保存的文件夹
        :param traindataset_path: 训练集路径
        :param valdataset_path: 验证集路径
        :param batchsize:
        '''

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #    
        self.device_count = torch.cuda.device_count()
        self.net = torch.nn.DataParallel(net.to(self.device), device_ids=range(torch.cuda.device_count()))
        self.save_path = save_path
        self.traindataset_path = traindataset_path
        self.valdataset_path = valdataset_path
        self.loss_func = loss_func
        self.lr = 2e-4
        self.Adam = optim.Adam(self.net.parameters(), lr=self.lr)
        self.SGD = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        self.batchsize = batchsize * self.device_count
        self.log = log
        if os.path.exists(self.save_path):
            net = nn.DataParallel(net)
            net.module.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")

    def trainer(self, round_limit=10):
        '''
        :param round_limit: 多少轮不更新，就停止训练
        :return:
        '''

        train_img = os.path.join(self.traindataset_path, 'JPEGImages')
        train_label = os.path.join(self.traindataset_path, 'SegmentationClass')
        val_img = os.path.join(self.valdataset_path, 'JPEGImages')
        val_label = os.path.join(self.valdataset_path, 'SegmentationClass')

        TrainDataset = MyDataset(train_img, train_label)
        valDataset = MyDataset(val_img, val_label)
        # batch_size一般不要超过百分之一 经验值
        train_dataloader = DataLoader(TrainDataset, self.batchsize, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(valDataset, self.batchsize, shuffle=True, num_workers=4)

        epoch = 1
        log = open(self.log, 'w', encoding='utf-8')
        log.write('epoch' + '\t' + 'loss' + '\t' + 'pa' + '\t' + 'iou' + '\t' + 'precision' + '\n')
        log.flush()

        flag = 0
        max_iou = 0
        bestepoch = 0
        min_loss = 100
        while epoch < 500:
            print('epoch - {} - training'.format(epoch))
            self.net.train()
            TP = FP = TN = FN = 0
            train_loss = 0
            batch = len(train_dataloader)
            for i, (img, mask) in enumerate(train_dataloader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                out = self.net(img)
                loss = self.loss_func(mask, out)

                self.Adam.zero_grad()
                loss.backward()
                self.Adam.step()

                if i % 10 == 0:
                    print('{}: {}/{} - loss: {}'.format(epoch, i, batch, loss.item()))
                    # torch.save(net.state_dict(), weight)
                    # print('save success')
                train_loss += loss.item()
            epoch_loss = train_loss / len(train_dataloader)

            print('epoch - {} - epoch_loss: {}'.format(epoch, epoch_loss))
            print('epoch - {} - evaluating'.format(epoch))

            self.net.eval()
            for img, mask in val_dataloader:
                img = img.to(self.device)
                mask = mask.to(self.device)
                with torch.no_grad():
                    pred =  self.net(img)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0

                TP += ((pred == 1) & (mask == 1)).cpu().sum().item()
                TN += ((pred == 0) & (mask == 0)).cpu().sum().item()
                FN += ((pred == 0) & (mask == 1)).cpu().sum().item()
                FP += ((pred == 1) & (mask == 0)).cpu().sum().item()

            pa = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FN)
            iou = TP / (TP + FP + FN)

            print('pa: ', pa)
            print('iou: ', iou)
            print('precision', precision)
            log.write(
                str(epoch) + '\t' + str(epoch_loss) + '\t' + str(pa) + '\t' + str(iou) + '\t' + str(precision) + '\n')
            log.flush()
            #取得是loss下降过程中最大iou
            if iou > max_iou:
                max_iou = iou
                bestepoch = epoch
                torch.save(self.net.module.state_dict(), self.save_path)
                # torch.save(self.net.state_dict(), self.save_path)
                print("保存成功，iou 更新为: {}".format(iou))
                flag = 0
            if epoch_loss < min_loss :
                min_loss = epoch_loss
                print("损失降低到了{}".format(min_loss))
                flag = 0
            else:
                flag += 1
                print("loss和iou都没变好，loss为{},iou为{}，第{}次未更新".format(min_loss,max_iou, flag))
                if flag == int(round_limit/2):
                    self.lr *= 0.2
                    print("学习率变为了{}".format(self.lr))
                if flag >= round_limit or self.lr < 5e-7:
                    print("early stop at epoch {}, finally iou: {}".format(bestepoch, max_iou))
                    break
            epoch += 1
        log.write(
            str(bestepoch) + '\t' + str(max_iou) +'\n')
        log.flush()
        log.close()



