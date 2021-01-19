import cv2
import os


#切割后的图片会出现全黑，影像训练，这个代码是删除全黑的
label_dir = os.listdir(r"C:\Users\lieweiai\Desktop\test\label")
print(label_dir)
for label in label_dir:
    name = label.split(".")[0]
    labelimg = cv2.imread(os.path.join(r"C:\Users\lieweiai\Desktop\test\label", label),0)
    labelimg[labelimg>200] = 255.0
    labelimg[labelimg <50] = 0.0
    min,max,minLoc,maxLoc=cv2.minMaxLoc(labelimg)
    if max ==0:
        os.remove(os.path.join(os.path.join(r"C:\Users\lieweiai\Desktop\test\label", label)))
        os.remove(os.path.join(r"C:\Users\lieweiai\Desktop\test\images" ,name+'.jpg'))



