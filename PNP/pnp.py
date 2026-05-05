import cv2
import numpy as np
mtx=np.array([[],[], [0, 0, 1]], dtype=np.float32)
dist=np.array([0],dtype=np.float32)
pattern_size = (9,6)
square_size = 25.0
objp=np.zeros((54,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
objp=objp*square_size
img=cv2.imread('.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners=cv2.findChessboardCorners(gray,pattern_size,None)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
success,rvec,tvec=cv2.solvePnP(objp,corners,mtx,dist)
R,_=cv2.Rodrigues(rvec)
print("旋转矩阵 R:\n",R)
print("平移向量 t:\n",tvec)
print("距离:",np.linalg.norm(tvec),"mm")