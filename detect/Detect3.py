import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from utils import saveImg
from test_model.D_Linknet import DinkNet50
from PIL import Image

import cv2
import os
from tqdm import tqdm


# n_classes = 2
# input_height = 256
# input_width = 256

# module = r"train_model\weight\5k_DlinkNet34.pth"
# module = r"train_model\weight2\u_net24k.pt"
# module = r"./train_model/weight/u_net.pt"
# module = r"./train_model/weight/U2NET.pt"
# module = r"./train_model/weight/UNet_2Plus.pt"
# module = r"./train_model/weight2/U2NET24k.pt"


# t0 = time.time()
# print("正在预测···共%s张" % len(list_images_path))


def predict_whole_picture(image1, step=128):
    module = r"D:\PycharmProjects(2)\2020-12-18-Deployment_model\test_model\DinkNet5024k.pt"

    out_path = "../test_model/save_images/"  # 预测之后上色保存的文件夹

    image_size = 2 * step
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = DinkNet34().cuda()
    net = DinkNet50().to(device)
    # net = u_net().cuda()
    # net = U2NET().cuda()
    # net = UNet_2Plus().cuda()

    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(module), False)

    net.eval()
    stride = image_size

    image_list = []
    image2 = image1
    predict_list = []

    img = cv2.cvtColor(np.asarray(image2), cv2.COLOR_RGB2BGR)
    image_h, image_w, _ = img.shape

    # print(image_h, image_w)  # 1024 1024   341, 584

    # 根据输入图片的大小不同，定义的矩阵的宽、高尺寸也不一样。
    if (image_h % stride == 0) and (image_w % stride != 0):
        padding_h = image_h
        padding_w = (image_w // stride + 1) * stride
    elif (image_h % stride != 0) and (image_w % stride == 0):
        padding_h = (image_h // stride + 1) * stride
        padding_w = image_w

    elif (image_h % stride == 0) and (image_w % stride == 0):
        padding_h = image_h
        padding_w = image_w
    elif (image_h % stride != 0) and (image_w % stride != 0):
        padding_h = (image_h // stride + 1) * stride
        # print(padding_h)  # 512
        padding_w = (image_w // stride + 1) * stride
        # print(padding_w)  # 768

    # padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8) + 255
    padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)

    # 在零矩阵三通道填充指定尺寸的图片
    padding_img[0:image_h, 0:image_w, :] = img[:, :, :]

    # 定义一个形状类似为（1152, 1152, 3）的白色图片矩阵
    padding_img1 = np.zeros((padding_h + step, padding_w + step, 3), dtype=np.uint8)

    padding_img1[(step // 2):(step // 2 + padding_h), (step // 2):(step // 2 + padding_w), :] = padding_img[:, :, :]

    # count_a = 1
    for h in range(padding_h // step):  # 1024 // 128 = 8
        for w in range(padding_w // step):  # 1024 // 128 = 8
            # 在填充后的图片进行划窗采样.(会有重叠区域采样),一共滑动8*8次。
            image_sample = padding_img1[(h * step):(h * step + stride),
                           (w * step):(w * step + stride), :]
            image_list.append(image_sample)
    # print(len(image_list))  # 一张大的原图滑动采样出64张图片

    # 对每个图像块进行预测
    count_b = 1
    for image in image_list:
        # print(np.shape(image))  # (256, 256, 3)
        img = transform(image).unsqueeze(0).to(device)
        # print(img.shape)  # torch.Size([1, 3, 256, 256])

        img = net(img)
        # print(img)

        img[img >= 0.5] = 1  # 输出的图片只有黑白两色
        img[img < 0.5] = 0

        img = img.cpu().detach()
        img = img.squeeze(0)
        # print(img.shape)  # torch.Size([1, 256, 256])

        # 保存网络输出的图片
        # img = img[0]

        img2 = img[:, (step // 2):(step // 2 + step), (step // 2):(step // 2 + step)]
        # print(np.shape(img2))

        img2 = img2[0]

        predict_list.append(img2)
    # print(len(predict_list))  # 64

    # 将预测后的图像块拼接起来
    tmp = torch.ones([padding_h, padding_w])

    for h in range(padding_h // step):  # for h in range(h_step):  8
        for w in range(padding_w // step):  # for h in range(h_step):  8
            # tmp[h * stride:(h + 1) * stride,w * stride:(w + 1) * stride] = predict_list[h * w_step + w]
            # 每走step, 就在定义的矩阵中填充网络所输出的对应值
            tmp[(h * step):(h * step + step), (w * step):(w * step + step)] = predict_list[
                h * (padding_h // step) + w]  # 4 * 1024 // 256 + 4

            # print(tmp.shape)  # (1024, 1024)

    path1 = r"D:\PycharmProjects(2)\2020-12-18-Deployment_model\test_model\test_images"
    save_image(tmp, os.path.join(path1, '{}.png'.format(1)))
    # img_out = transforms.ToPILImage()(tmp).convert('RGB')

    abs_path = os.path.join(path1, '{}.png'.format(1))

    return abs_path


if __name__ == '__main__':
    print("-----------正在预测整副图像-----------")
    list_images_path = r"D:\PycharmProjects(2)\2020-12-18-Deployment_model\test_model\test_images\1.jpg"
    image = cv2.imread(list_images_path)
    step = 256  # 128
    image_size = 2 * step

    predict_whole_picture(image, step)
    # t1 = time.time()
    print("----------预测完成-----------")
    #    print("耗时", str((t1 - t0) * 1000), "ms")
    # print("耗时：", '%.2f' % ((t1 - t0) / 60), "min")
