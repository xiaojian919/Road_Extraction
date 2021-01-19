import os
from matplotlib import pyplot as plt

PATH = r"D:\lieweicodetest\Road_Extraction\train_model\log\DinkNet5024k.txt"
name = PATH.split("\\")[-1].split(".")[0]
epoch = []
loss = []
pix_acc = []
iou = []
precision =[]
numname = ["loss","pix_acc","iou","precision"]
num = [loss,pix_acc,iou,precision]
with open(PATH, 'r') as f1:
    list1 = f1.readlines()
    print(len(list1[1:]))
    for line in list1[1:]:
        strs = line.split()
        if len(strs)==5:
            epoch.append(int(strs[0]))
            loss.append(float(strs[1]))
            pix_acc.append(float(strs[2]))
            iou.append(float(strs[3]))
            precision.append(float(strs[4]))
    for i in range(4):
        x = epoch
        y = num[i]
        plt.title("{},epoch-{}".format(name,str(numname[i])))
        plt.plot(x, y)
        # plt.savefig("filename.png")
        plt.show()




