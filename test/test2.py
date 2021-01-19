import cv2


path = r"C:/Users/lieweiai/Desktop/detectimg/img/297_sat.jpg"
image = cv2.imread(path,1)
w,h,_ = image.shape
print(w,h)

cv2.imshow('image', image)  # 展示图片
cv2.waitKey(0)
cv2.destroyAllWindows()