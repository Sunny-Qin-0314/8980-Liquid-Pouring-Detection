import numpy as np
import cv2 as cv
from math import *
from sklearn.cluster import DBSCAN

cap = cv.VideoCapture('test4.mov')
fgbg = cv.createBackgroundSubtractorMOG2()
writer = None
while(1):
    (ret, frame)= cap.read()
    if not ret:
		break
    #frame = cv.rotate(frame, rotateCode=cv.ROTATE_90_CLOCKWISE)
    median = cv.medianBlur(frame,5)
    new=cv.GaussianBlur(median,(5,5),0)
    fgmask = fgbg.apply(new)
    # Sobel
    grad_x = cv.Sobel(fgmask, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(fgmask, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    thresh = 150
    idx = np.where(grad > thresh)
    coor = zip(idx[0], idx[1])
    Z = np.asarray(coor)
    # print("Z",Z.shape)

    # convert to np.float32
    Z = np.float32(Z)
    cv.imshow('frame', frame)
    cv.imshow('grad', grad)
    # if Z.shape > 0:
    #     # print(Z.shape)
    #     '''
    #     # define criteria, number of clusters(K) and apply kmeans()
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #     K = 2
    #     rett,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    #     center = np.uint8(center)
    #     res = center[label.flatten()]
    #     res2 = res.reshape((grad.shape))
    #     print(center)
    #
    #     #clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    #     frame = cv.circle(frame,(center[0,0],center[0,1]), 5, (0,0,255), -1)
    #     frame = cv.circle(frame,(center[1,0],center[1,1]), 5, (255,0,0), -1)
    #     # frame = cv.circle(frame,(center[2,0],center[2,1]), 5, (0,255,0), -1)
    #     '''
    #     # cv.imshow('frame', frame)
    #     # cv.imshow('grad', grad)
    #     #cv.imshow('res2',res2)
    print(fgmask.shape)
    # if writer is None:
	# 	# initialize our video writer
	# 	fourcc = cv.VideoWriter_fourcc(*"MJPG")
	# 	writer = cv.VideoWriter('output.avi', fourcc, 30,(fgmask.shape[1], fgmask.shape[0]), True)
    #
    #
	# # write the output frame to disk
    # writer.write(fgmask)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# writer.release()
cap.release()
cv.destroyAllWindows()
