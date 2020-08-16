import cv2 as cv
import numpy as np

def track_fluid(frame):
    
    # Sobel
    grad_x = cv.Sobel(frame, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(frame, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # reshape the gradient array into vector
    # if the size is not even, have trouble reshaping into 2 columns array
    # so that add zero at the end
    flatt_grad = grad.flatten()
    gd_v = flatt_grad
    if (flatt_grad.size)%2 == 0:
        gd_v = grad.reshape(-1,2)
    else:
        gd_v = np.append(gd_v, 0).reshape((-1,2)) # add zero at the end
        
    # convert to float values
    gd_v = np.float32(gd_v)

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    K = 2
    rett,label,center = cv.kmeans(gd_v, K, None, criteria, 10, flags)
    center = np.uint8(center)
    res = center[label.flatten()]

    # if added zero for reshaping evenly, delete the last value which is zero
    if (flatt_grad.size)%2 != 0:
        print("here")
        res = np.delete(res.flatten(), res.flatten().size-1)

    res2 = res.reshape((grad.shape))
    #print("center",center)
    #print(res[30:50])
    #print(np.amax(center))
    #print(grad.shape)

    return grad, label, center





