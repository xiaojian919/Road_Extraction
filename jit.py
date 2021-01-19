import torch
import cv2
import torch.nn.functional as F
from Models import u_net
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    print(torch.__version__)
    exit()
    save_params = r'C:/Users/Desktop/result/u_net24k.pt'
    device = torch.device('cpu')  # 使用gpu进行推理

    net = u_net().to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(save_params), False)  # 加载模型
    net.eval()  # 把模型转为test模式

    # img = cv2.imread(r"C:/Users/lieweiai/Desktop/detectimg/img/113_sat_00.jpg", 1)  # 读取要预测的灰度图片
    # img = Image.fromarray(img)
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    # img = trans(img)
    # img = img.unsqueeze(0)  # 图片扩展多一维,[batch_size,通道,长，宽]
    # img = img.to(device)
    # print(np.shape(img))
    img = torch.rand(1, 3, 256, 256).to(device)

    traced_net = torch.jit.trace(net, img)  # 打包：记录了整个推理的一个路径流
    traced_net.save("model.pt")
    print("模型序列化导出成功")

