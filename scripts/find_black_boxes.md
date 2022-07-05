# Find black boxes
Algorithm intends to identify the black field within the gasmeter in form of its 4 corner points.

1. Convert BGR image to grayscale, while ignoring the red color.
2. `cv.threshold()` with fixed threshold starting at 50 and increase by 10 if no black box was found until 100.
3. `cv.findContours()`
4. Filter contours > 10e3 px.
5. Apply `cv.convexHull()` to the contours.
6. Find smallest possible rectangle using `cv.minAreaRect()` 
7. Calculate aspect ratio of the minRect:  `max(h/w, w/h)`
8. Filter contours for aspect ratio in range around 8.83
9. Filter contours for area being >95% of area minRext.
10. The contour that passed all filters must be approximated down to 4 points. 


