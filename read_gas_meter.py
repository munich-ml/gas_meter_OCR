import cv2

sample = cv2.imread('gas_sample.jpg', cv2.IMREAD_COLOR)

if True:
    cv2.imshow("sample", sample)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()