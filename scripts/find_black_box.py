# %% FIND BLACK BOX SCRIPT
import numpy as np
import cv2 as cv
import json, os, pytesseract, time
import datetime as dt


def bgr_2_gray(img):
    """ Converts a BGR image into a grayscale image based on blue and green, ignoring red 
    """
    return np.array(img[:,:,:2].mean(axis=2), dtype=np.uint8)


def approx_contour(contour, n_points=4, max_iteration=1000):
    """Approcimates a opencv contour down to n_points using cv.approxPolyDP() 

    Args:
        contour (contour): contour to be approximated
        n_points (int, optional): Desired number of points for the returned contour. Defaults to 4.
        max_iteration (int, optional): Don't try forever. Defaults to 1000.

    Returns:
        Contour / or None: Approximated countour
    """
    fract = 0.1            # fraction of the perimeter as a measure for approximation detail
    fract_mod = 2.0        # modify fract by this factor in the next trial
    for _ in range(max_iteration):
        apx = cv.approxPolyDP(contour, fract * cv.arcLength(contour, True), True)
        if len(apx) > n_points:   # More than n_point approximation: Too detailed!
            fract *= fract_mod
        elif len(apx) < n_points:   # Less than n_point approximation: Too crude
            fract /= fract_mod
            fract_mod *= 0.9  # safety measure: Reduce step size to exclude oscillation around a small target
        elif len(apx) == n_points:
            return apx 
    
    raise RuntimeError("Waring: Approximation to 4 points failed")
    

    
def normalize_rectangle(contour):
    """ Normalized an opencv contour in a way that the points are in the same order.

    Arg: contour (np.array) 

    Returns: np.array: Normalized contour
    """
    def calc_distance(p1, p2):
        """Calculate distance between two points
        p1 and p2 in format (x1,y1) and (x2,y2) tuples
        """
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    def calc_middle(line):
        """Calculate center point in a line
        """
        p1, p2 = line
        return [int((xy1 + xy2) / 2) for xy1, xy2 in zip(p1[0], p2[0])]
    
    p1, p2, p3, p4 = contour
    if calc_distance(p1[0], p2[0]) < calc_distance(p2[0], p3[0]):
        edge1, edge2 = [p1, p2], [p3, p4]
    else:
        edge1, edge2 = [p1, p4], [p2, p3]
        
    if calc_middle(edge1)[0] < calc_middle(edge2)[0]:
        upper_left, lower_left = edge1
        lower_right, upper_right = edge2
    else:
        upper_left, lower_left = edge2
        lower_right, upper_right = edge1
        
    if upper_left[0][1] < lower_left[0][1]:
        upper_left, lower_left = lower_left, upper_left
    
    if upper_right[0][1] < lower_right[0][1]:
        upper_right, lower_right = lower_right, upper_right
        
    return np.array([upper_left[0], lower_left[0], lower_right[0], upper_right[0]], dtype=np.float32)



def find_black_number_field(img_bgr, numfield_width=1000):
    img_gray = bgr_2_gray(img_bgr)

    # try different grayscale contours for number field search
    grayscale_thresholds=[50, 60, 70, 80, 90, 100, 110]
    for threshold in grayscale_thresholds:
        # Apply binary fixed threshold
        _, img_th = cv.threshold(img_gray, thresh=threshold, maxval=255, type=cv.THRESH_BINARY)
        
        # find contours
        contours, _ = cv.findContours(img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # ignore tiny contours
        big_contours = [c for c in contours if cv.contourArea(c) > 1e3]
        
        # force convex contours
        convex_contours = [cv.convexHull(c) for c in big_contours]
        
        # find the right gasmeter contour    
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
            try:
                apx = approx_contour(contour)
            except RuntimeError as e:
                print(e)
                continue
            
            
            w2, h2 = numfield_width, int(numfield_width / 9.2)  # aspect ratio of number field
            pts2 = np.float32([[0, h2], [0, 0], [w2, 0], [w2, h2]])
            cnorm = normalize_rectangle(apx)

            matrix = cv.getPerspectiveTransform(cnorm, pts2)
            img_number_field = cv.warpPerspective(img_bgr, matrix, (w2, h2))
            
            area = cv.contourArea(contour)
            params = {"threshold": threshold, "area": area, "ascpect_ratio": aspect_ratio, "rect_factor": rect_factor,
                      "numfield_width": numfield_width}

            return img_number_field, contour, params
            
        

if __name__ == "__main__":
    
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('m','j','p','g'))
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    
    store_results = True
    
    while True:
        _, img_bgr = cap.read()
        number_field = find_black_number_field(img_bgr=img_bgr, numfield_width=1000)
        if number_field is not None:
            img_number_field, contour, params = number_field
            
            # OCR
            text = pytesseract.image_to_string(bgr_2_gray(img_number_field))
            ocr_plausible = False
            if len(text) >= 8:
                if text[:8].isdecimal():
                    ocr_plausible = True
                    params["text"] = text
            params["ocr_plausible"] = ocr_plausible
            
            # save results to files
            if store_results:
                fpb = os.path.join("results", dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-4])
                with open(fpb+"_params.json", "w") as file:
                    json.dump(params, file)
                cv.imwrite(fpb +"_img_bgr.jpg", img_bgr)
                cv.imwrite(fpb +"_img_number_field.jpg", img_number_field)
            
            # show results on screen
            cv.imshow("Digits", img_number_field)    
            color = {True: (0, 255, 0), False: (0, 0, 255)}[ocr_plausible]
            cv.drawContours(img_bgr, [contour], -1, color, 2)
            
        cv.imshow("BGR", img_bgr)

        if cv.waitKeyEx(10) & 0xFF == ord('q'):
            break
        
        time.sleep(0.01)

    cap.release()
    cv.destroyAllWindows()
     
                    
            
