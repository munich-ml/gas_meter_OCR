# %% FIND BLACK BOX SCRIPT
import numpy as np
import cv2 as cv


def bgr_2_gray(img):
    """ Converts a BGR image into a grayscale image based on blue and green, ignoring red 
    """
    return np.array(img[:,:,:2].mean(axis=2), dtype=np.uint8)


if __name__ == "__main__":
    
    cap = cv.VideoCapture(0)

    file = open("black_boxes.csv", "w") 
    file.write("threshold, area, aspect_ratio, rect_factor\n")
                
    while True:
        _, img_bgr = cap.read()
        img_gray = bgr_2_gray(img_bgr)

        for threshold in [50, 60, 70, 80, 90, 100, 110]:
            # Apply binary fixed threshold
            _, img_th = cv.threshold(img_gray, thresh=threshold, maxval=255, type=cv.THRESH_BINARY)
            
            # find contours
            contours, _ = cv.findContours(img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            # ignore tiny contours
            big_contours = [c for c in contours if cv.contourArea(c) > 1e3]
            
            # force convex contours
            convex_contours = [cv.convexHull(c) for c in big_contours]
            
            # find the right gasmeter contour    
            gasmeter_found = False
            for contour in convex_contours:
                minRect = cv.minAreaRect(contour)
                (x, y), (h, w), angle = minRect
                
                # Test 1: Aspect ratio:
                aspect_ratio = max(h/w, w/h)
                if not(7 < aspect_ratio < 11):
                    continue
                    
                # Test2: How much is the contour a rectangle
                rect_factor = cv.contourArea(contour) / (h*w)
                if rect_factor < 0.91:
                    continue
                
                # Only the right gasmeter contours get to this point
                # Now reduce the contour down to 4 points (but not less)
                area = cv.contourArea(contour)
                print(f"Contour: {threshold=}, {area=}, {aspect_ratio=}, {rect_factor=}")
                file.write(f"{threshold}, {area}, {aspect_ratio}, {rect_factor}\n")
                cv.drawContours(img_bgr, [contour], -1, (0, 255, 0), 2)
                gasmeter_found = True
                break
                
            if gasmeter_found:
                break
            
                
        cv.imshow("BGR", img_bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    file.close()
    cap.release()
    cv.destroyAllWindows()
     
                    
            
