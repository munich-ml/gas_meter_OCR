import cv2

img = cv2.imread('gas_sample.jpg', cv2.IMREAD_COLOR)

def rescale_image(image, scale=0.75):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

#cv2.imshow("sample image", img)    

img_mini = rescale_image(img, 0.2)

cv2.imshow("rescaled image", rescale_image(img, 0.2)) 
   
cv2.waitKey(0)
cv2.destroyAllWindows()