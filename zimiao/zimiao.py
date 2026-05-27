import cv2
import numpy as np
import time
import sys
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
CAMERA_MATRIX=np.array([[568.18492659,0.0,270.70902258],[0.0,564.52638288,238.80761855],[0.0,0.0,1.0]],dtype=np.float32)
DIST_COEFFS=np.array([-1.92828805e-01,1.19153597e+00,1.20612618e-03,-1.54875602e-02,-2.19634037e+00],dtype=np.float32)
def find_armor(image,camera_matrix,dist_coeffs):
    ARMOR_WIDTH=0.225
    ARMOR_HEIGHT=0.055
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3),np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lights=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if h>w*1.5 :
            area=cv2.contourArea(contour)
            if area<10:
                continue
            rect=cv2.minAreaRect(contour)
            angle=rect[2]
            if w>h:
                w,h=h,w
                angle=90-angle
            lights.append((x,y,w,h,angle))
    armors=[]
    used=[False]*len(lights)
    for i in range(len(lights)):
        if used[i]:
            continue
        x1,y1,w1,h1,a1=lights[i]
        for j in range(i+1,len(lights)):
            if used[j]:
                continue
            x2,y2,w2,h2,a2=lights[j]
            if min(h1,h2)/max(h1,h2)<0.7:
                continue
            if x1<x2:
                left=lights[i]
                right=lights[j]
            else:
                right=lights[i]
                left=lights[j]
            gap=right[0]-(left[0]+left[2])
            if gap>150:
                continue
            if abs(left[4]-right[4])>15:
                continue
            left_top=(left[0],left[1])
            left_bottom=(left[0],left[1]+left[3])
            right_top=(right[0]+right[2],right[1])
            right_bottom=(right[0]+right[2],right[1]+right[3])
            image_points=np.array([left_top,right_top,right_bottom,left_bottom],dtype=np.float32)
            half_w=ARMOR_WIDTH/2.0
            half_h=ARMOR_HEIGHT/2.0
            object_points=np.array([[-half_w,-half_h,0],[half_w,-half_h,0],[half_w,half_h,0],[-half_w,half_h,0]],dtype=np.float32)
            success,rvec,tvec=cv2.solvePnP(object_points,image_points,camera_matrix,dist_coeffs)
            if not success:
                continue
            armors.append({'tvec':tvec.flatten()})
            used[i]=used[j]=True
            break
    return armors
CAM_TO_GIMBAL_TRANS=(0.05,0.0,0.08)
CAM_TO_GIMBAL_ROT=(0.0,0.0,0.0)
def compute_target_angles(target_cam):
  r=R.from_euler('xyz',CAM_TO_GIMBAL_ROT)
  target_gimbal=r.apply(target_cam)+np.array(CAM_TO_GIMBAL_TRANS)
  yaw=np.arctan2(target_gimbal[0],target_gimbal[2])
  pitch=np.arctan2(-target_gimbal[1],target_gimbal[2])
  return yaw,pitch  
DT=1/30.0
class PredictiveKalman:
    def __init__(self,dt=DT):
        self.dt=dt
        self.kf=KalmanFilter(dim_x=4,dim_z=2)
        self.kf.F=np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
        self.kf.H=np.array([[1,0,0,0],[0,0,1,0]])
        self.kf.P=np.eye(4)*100
        self.kf.R=np.eye(2)*0.05*0.05
        self.kf.Q=np.eye(4)*0.01*0.01
        self.kf.x=np.zeros(4)
        self.last_update_time=None
    def update(self,yaw_rad,pitch_rad):
        now=time.time()
        if self.last_update_time is not None:
            dt_real=now-self.last_update_time
            if dt_real>0:
                self.predict(dt=dt_real)
        self.kf.update([yaw_rad,pitch_rad])
        self.last_update_time=now
    def predict(self,dt=None):
        if dt==None:
            dt=self.dt
        F=np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
        self.kf.x=F@self.kf.x
        self.kf.P=F@self.kf.P@F.T+self.kf.Q
    def predict_future(self,future_time):
        steps=max(1,int(round(future_time/self.dt)))
        x_saved=self.kf.x.copy()
        P_saved=self.kf.P.copy()
        for i in range(steps):
            self.predict(dt=self.dt)
        yaw_pred=self.kf.x[0]
        pitch_pred=self.kf.x[2]
        return yaw_pred,pitch_pred
class TrackedTarget:
    def __init__(self,target_id,tvec):
        self.id=target_id
        self.tvec=tvec
        self.last_seen=time.time()
def main():
    if len(sys.argv)>1:
        source=sys.argv[1]
        if source.isdigit():
            source=int(source)
        cap=cv2.VideoCapture(source)
    else:
        cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error")
        return 
    kf=PredictiveKalman(dt=DT)
    print("start")
    current_target=None
    current_id=0
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        armors=find_armor(frame,CAMERA_MATRIX,DIST_COEFFS)
        if armors:
            best_match=None
            if current_target:
                best_match=min(armors,key=lambda a: np.linalg.norm(a['tvec']-current_target.tvec))
                if np.linalg.norm(best_match['tvec']-current_target.tvec)>0.5:
                    best_match=None
            if best_match:
                yaw_obs,pitch_obs=compute_target_angles(best_match['tvec'])
                kf.update(yaw_obs,pitch_obs)
                current_target.tvec=best_match['tvec']
                current_target.last_seen=time.time()
            else:
                target=min(armors,key=lambda a:a['tvec'][2])
                target_cam=target['tvec']
                yaw_obs,pitch_obs=compute_target_angles(target_cam)
                kf=PredictiveKalman(dt=DT)
                kf.update(yaw_obs,pitch_obs)
                current_target=TrackedTarget(current_id,target['tvec'])
                current_id+=1
        else:
            if current_target and (time.time()-current_target.last_seen)>1.0:
                current_target=None
            else:
                kf.predict(dt=DT)
        yaw_pred,pitch_pred=kf.predict_future(0.1)
        yaw_deg=np.degrees(yaw_pred)
        pitch_deg=np.degrees(pitch_pred)
        print(f"yaw:{yaw_deg:6.2f},pitch:{pitch_deg:6.2f}",end='',flush=True)
        cv2.imshow("slefaim",frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()