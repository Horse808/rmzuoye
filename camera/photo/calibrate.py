import cv2
import numpy as np
import glob
pattern_size=(9,6)#neibujiaodianshu
square_size=25.0    #gezichangdu(mm) 
objp=np.zeros((pattern_size[0]*pattern_size[1],3),np.float32)
objp[:,:2]=np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp=objp*square_size
objpoints=[]
imgpoint=[]
images=glob.glob('*.jpg')
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
for fname in images:
    img=cv2.imread(fname)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,pattern_size,None)
    if ret:
        corners_subpix=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
    imgpoint.append(corners_subpix)
ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objpoints,imgpoint,gray.shape[::-1],None,None)
print("mtx:\n",mtx)
print("dist:\n",dist.ravel())
print(f"\nwucha:{ret:.6f}")