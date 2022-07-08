# %% FIND BLACK BOX SCRIPT
import numpy as np
import cv2 as cv


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
    

def printc(contour):
    for item in contour:
        print(f"{item}, ", end="")
    print()
    
    
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


if __name__ == "__main__":
    
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('m','j','p','g'))
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    

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
                try:
                    apx = approx_contour(contour)
                except RuntimeError as e:
                    print(e)
                    continue
                
                w2, h2 = 1000, 110
                pts2 = np.float32([[0, h2], [0, 0], [w2, 0], [w2, h2]])
                cnorm = normalize_rectangle(apx)

                matrix = cv.getPerspectiveTransform(cnorm, pts2)
                img_dig = cv.warpPerspective(img_bgr, matrix, (w2, h2))
                
                area = cv.contourArea(contour)
                print(f"Contour: {threshold=}, {area=}, {aspect_ratio=}, {rect_factor=}")
                file.write(f"{threshold}, {area}, {aspect_ratio}, {rect_factor}\n")
                cv.drawContours(img_bgr, [contour], -1, (0, 255, 0), 2)
                gasmeter_found = True
                break
                
            if gasmeter_found:
                cv.imshow("Digits", img_dig)
                break
            
        cv.imshow("BGR", img_bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    file.close()
    cap.release()
    cv.destroyAllWindows()
     
                    
            
