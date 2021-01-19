from PIL import Image
from glob import glob
import random
from test.HSV import randomHueSaturationValue
import cv2

# im.transpose(Image.FLIP_LEFT_RIGHT)       #左右对换。
# im.transpose(Image.FLIP_TOP_BOTTOM)       #上下对换。
# im.transpose(Image.ROTATE_90)             #旋转 90 度角。
# im.transpose(Image.ROTATE_180)            #旋转 180 度角。
# im.transpose(Image.ROTATE_270)            #旋转 270 度角。


def change(imgspath, savepath):
    print("一共有{}张图".format(len(imgspath)))
    count = 0
    for imgpath in imgspath:
        print(imgpath)  # 图片的绝对路径

        name = imgpath.split("\\")[-1]
        picname = name.split(".")[0]  # 100034_sat_00
        pictype = name.split(".")[1]  # jpg

        img = Image.open(r"{}".format(imgpath))
        img.save(savepath + r"\{}.{}".format(picname, pictype))

        img1 = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        img1.save(savepath + r"\{}_LR2.{}".format(picname, pictype))

        img2 = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
        img2.save(savepath + r"\{}_TB.{}".format(picname, pictype))

        img3 = img.transpose(Image.ROTATE_90)  # 旋转90度
        img3.save(savepath + r"\{}_90.{}".format(picname, pictype))

        img4 = cv2.imread(imgpath)  # 做HSV操作
        img = randomHueSaturationValue(img4)
        cv2.imwrite(savepath + r"\{}_HSV.{}".format(picname, pictype), img)

        # 随机翻折：包含水平、竖直、对角线三种翻折方式
        # 随机缩放：将图像随机缩放至多10%。
        # 随机旋转：逆时针或顺时针旋转90度
        # 在HSV空间对图像进行色彩变换

        if count % 100 == 0:  # 每处理100张图片打印一次
            print(count)

        count += 1


if __name__ == '__main__':
    imgspath = glob(r"C:\Users\lieweiai\Desktop\test\images\*.jpg")
    savepath = r"C:\Users\lieweiai\Desktop\test\images"
    change(imgspath, savepath)



