from Models import *
from loss import *
from train import Trainer
import os

if __name__ == '__main__':
    net = DinkNet50()
    if not os.path.exists("./weight"):
        os.makedirs("./weight")
    save_path = r'.\weight\120k_DinkNet.pt'

    if not os.path.exists("./log"):
        os.makedirs("./log")
    log = r'.\log\120k_DinkNet_log.txt'

    loss = dice_bce_loss()
    traindataset_path = r'C:\Users\Desktop\120k\train_set'
    valdataset_path = r'C:\Users\Desktop\120k\validate_set'
    trainer = Trainer(net, loss_func=loss, save_path=save_path, traindataset_path=traindataset_path,
                      valdataset_path=valdataset_path, batchsize=2, log=log)

    trainer.trainer(10)