import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time



def draw_box_on_red_shapes(img):
    # Fokus on red color
    def fokus_red(img):
        bg = img[:,:,:2].mean(axis=2)
        red = img[:,:,2]
        reddish = np.maximum(red-bg, np.zeros_like(red))
        return np.array(reddish, dtype=np.uint8)

    img_red = fokus_red(img)
    
    # Thresholding
    _, img_thresh = cv.threshold(img_red, 0, 255, cv.THRESH_OTSU)

    contours, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Contour simplification
    hulls = [cv.convexHull(contour) for contour in contours]

    approximations = []
    for hull in hulls:
        epsilon = 0.1 * cv.arcLength(hull, True)
        apx = cv.approxPolyDP(hull, epsilon, True)
        approximations.append(apx)

    cv.drawContours(img, approximations, -1, (0, 255, 0), 2)

    img = cv.putText(img, text=f"{img_red.max()=}", org=(10, 60), 
                    fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, 
                    color=(0, 255, 0), thickness=2, bottomLeftOrigin=False)

    return img, img_thresh, img_red

if __name__ == "__main__":
    
    cap = cv.VideoCapture(0)
    t0 = time.time()
    fps = []
    
    while True:
        _, bgr = cap.read()
        reds, img_thresh, img_red = draw_box_on_red_shapes(bgr)

        fps.append(1/(time.time()-t0))
        if len(fps)==20:
            mean_fps = np.array(fps).mean()
            fps = []
        t0 = time.time()
        
        if "mean_fps" in dir():
            reds = cv.putText(reds, text=f"rate = {mean_fps :.0f} fps", org=(10, 30), 
                            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, 
                            color=(0, 255, 0), thickness=2, bottomLeftOrigin=False)

        cv.imshow("Red shape detector", reds)
        cv.imshow("Thresholded image", img_thresh)
        cv.imshow("red - mean(green, blue)", img_red)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
