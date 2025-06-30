import cv2


img = cv2.imread('../assets/object1.png',0)
img = cv2.resize(img,(0,0),fx=5,fy=5)
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# create new image
# cv2.imwrite('../assets/img1.png',img)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

