import numpy as np
import torch

#这个是验证混淆矩阵的
a = torch.tensor(np.random.randint(0,2,9).reshape([1,3,3]))
b = torch.tensor(np.random.randint(0,2,9).reshape([1,3,3]))
# TP = (a * b).sum()
print(a)
print(b)

# prediction = torch.argmax(a,dim=0)
TP =((a ==1)&(a == 1)).cpu().sum()
TN =((a ==0)&(a == 0)).cpu().sum()
FN =((a ==0)&(a == 1)).cpu().sum()
FP =((a ==1)&(a == 0)).cpu().sum()
# total+=prediction.eq(target).cpu().sum()
print(TP)
print(TN)
print(FN)
print(FP)
print(TP+TN+FN+FP)


# img_path = r'E:\PyCharmProject\datasets\patch\image'
# mask_path = r'E:\PyCharmProject\datasets\patch\mask'
img_path = r'E:\PyCharmProject\datasets\5k\train_set\JPEGImages'
mask_path = r'E:\PyCharmProject\datasets\5k\train_set\SegmentationClass'
