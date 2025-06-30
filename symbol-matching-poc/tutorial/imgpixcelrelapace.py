import cv2
import random

img = cv2.imread('../assets/object1.png',-1)

# change img pixcel
# for i in range(10):
#     for j in range(img.shape[1]):
#         img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(0,255)]

# cut the image pixcel to place another position
tag = img[50:70,60:90]
img[10:30,60:90] = tag


cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




