from Models import *
from loss import *
from train import Trainer
import os

if __name__ == '__main__':
    net = U2NET()
    if not os.path.exists("./weight"):
        os.makedirs("./weight")
    save_path = r'.\weight\120k_u2net.pt'

    if not os.path.exists("./log"):
        os.makedirs("./log")
    log = r'.\log\120k_u2net_log.txt'

    loss = dice_bce_loss()
    traindataset_path = r'C:\Users\Desktop\120k\train_set'
    valdataset_path = r'C:\Users\Desktop\120k\validate_set'
    trainer = Trainer(net, loss_func=loss, save_path=save_path, traindataset_path=traindataset_path,
                      valdataset_path=valdataset_path, batchsize=4, log=log)

    trainer.trainer(10)