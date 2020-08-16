import numpy as np
import cv2 as cv
from math import *
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    # Background subtractor
    fgmask = fgbg.apply(frame)
    
    # Sobel
    grad_x = cv.Sobel(fgmask, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(fgmask, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    #cv.imshow('frame_fg',fgmask)
    #cv.imshow('frame_gradient', grad)

    # reshape the gradient array into vector
    gd_v = grad.reshape((-1,2))
    gd_v = np.float32(gd_v)

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    K = 2
    rett,label,center = cv.kmeans(gd_v, K, None, criteria, 10, flags)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((grad.shape))
    print("center",center)
    print(res[30:50])
    print(np.amax(center))
    print(grad.shape)
    
    # drawing clusters on frames
    frame = cv.circle(frame,(center[0,0],center[0,1]), 5, (0,0,255), -1)
    frame = cv.circle(frame,(center[1,0],center[1,1]), 5, (255,0,0), -1)
    cv.imshow('frame', frame)
    cv.imshow('res', res2)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv.destroyAllWindows()
