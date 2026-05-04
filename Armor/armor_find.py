import cv2
import numpy as np
import glob
def find_armor(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3),np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    contours,b=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    light=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if h>w*1.5 and h>30 and w<50:
            area=cv2.contourArea(contour)
            angle=cv2.minAreaRect(contour)[2]
            if w>h:
                w,h=h,w
                angle=90-angle
            if area>100:
                light.append((x,y,w,h,angle))
    armors=[]
    used=[False]*len(light)
    for i in range(len(light)):
        if used[i]:
            continue
        x1,y1,w1,h1,a1=light[i]
        for j in range(i+1,len(light)):
            if used[j]:
                continue
            x2,y2,w2,h2,a2=light[j]
            if min(h1,h2)/max(h1,h2)<0.7:
                continue
            if x1<x2:
                left,right=(x1,y1,w1,h1),(x2,y2,w2,h2)
            else:
                right,left=(x1,y1,w1,h1),(x2,y2,w2,h2)
            gap=right[0]-(left[0]+left[2])
            if gap<20 or gap>150:
                continue
            if abs(a1-a2)>15:
                continue
            left_x=left[0]
            right_x=right[0]+right[2]
            top_y=min(left[1],right[1])
            bottom_y=max(left[1]+left[3],right[1]+right[3])
            armors.append({'center':((left_x+right_x)//2,(top_y+bottom_y)//2),'bbox':(left_x-10,top_y-10,right_x-left_x+20,bottom_y-top_y+20),'llight':left,'rlight':right})
            used[i]=used[j]=True
            break
    return armors,binary
if __name__=="__main__":
    images=glob.glob('*.jpg')
    for img_path in images:
        img=cv2.imread(img_path)
        if img is not None:
            armors,binary=find_armor(img)
            result=img.copy()
            for armor in armors:
                x,y,w,h=armor['bbox']
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(result,armor['center'],5,(0,0,255),-1)
                lx,ly,lw,lh=armor['llight']
                rx,ry,rw,rh=armor['rlight']
                cv2.rectangle(result,(lx,ly),(lx+lw,ly+lh),(255,0,0),2)
                cv2.rectangle(result,(rx,ry),(rx+rw,ry+rh),(255,0,0),2)
                cv2.imshow("binary",binary)
                cv2.imshow("result",result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    cap=cv2.VideoCapture('avi.mp4')
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        armors,_=find_armor(frame)
        for armor in armors:
            x,y,w,h=armor['bbox']
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('video',frame)
        if cv2.waitKey(60) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()