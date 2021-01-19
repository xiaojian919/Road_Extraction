from Models import *
from loss import *
from train import Trainer
import os

if __name__ == '__main__':
    net = U2NET()
    if not os.path.exists("./weight"):
        os.makedirs("./weight")
    save_path = r'.\weight\5k_u2net.pt'

    if not os.path.exists("./log"):
        os.makedirs("./log")
    log = r'.\log\5k_u2net_log.txt'

    loss = dice_bce_loss()
    traindataset_path = r'C:\Users\Desktop\test_img3'
    valdataset_path = r'C:\Users\Desktop\test_img3'
    trainer = Trainer(net, loss_func=loss, save_path=save_path, traindataset_path=traindataset_path,
                      valdataset_path=valdataset_path, batchsize=1, log=log)

    trainer.trainer(10)