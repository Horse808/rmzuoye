import cv2
import numpy as np
import sys
import time

def test_lights_detection(source):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    print("灯条测试程序启动")
    current_display = 3
    threshold_value = 40
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lights = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w * 0.5:
                area = cv2.contourArea(cnt)
                if area < 5: continue
                rect = cv2.minAreaRect(cnt)
                angle = rect[2]
                if w > h:
                    w, h = h, w
                    angle = 90 - angle
                #if abs(angle) > 80: continue
                lights.append((x, y, w, h, angle))
        
        # 简单匹配装甲板
        armors = []
        used = [False] * len(lights)
        for i in range(len(lights)):
            if used[i]: continue
            x1, y1, w1, h1, a1 = lights[i]
            for j in range(i+1, len(lights)):
                if used[j]: continue
                x2, y2, w2, h2, a2 = lights[j]
                if min(h1, h2)/max(h1, h2) < 0.7: continue
                if x1 < x2:
                    left, right = lights[i], lights[j]
                    li, ri = i, j
                else:
                    left, right = lights[j], lights[i]
                    li, ri = j, i
                gap = right[0] - (left[0] + left[2])
                gap_thresh = max(40, (left[3] + right[3]) * 1.5)
                if gap > gap_thresh: continue
                if abs(left[4] - right[4]) > 20: continue
                # 匹配成功
                bbox = (min(left[0], right[0]),
                        min(left[1], right[1]),
                        max(left[0]+left[2], right[0]+right[2]) - min(left[0], right[0]),
                        max(left[1]+left[3], right[1]+right[3]) - min(left[1], right[1]))
                center = ((left[0]+left[2]//2 + right[0]+right[2]//2)//2,
                          (left[1]+left[3]//2 + right[1]+right[3]//2)//2)
                armors.append({'bbox': bbox, 'center': center})
                used[li] = used[ri] = True
                break
        
        # 显示
        display = frame.copy()
        if current_display == 1:
            display = frame
        elif current_display == 2:
            display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        elif current_display == 3:
            for (x, y, w, h, _) in lights:
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for a in armors:
                x, y, w, h = a['bbox']
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(display, a['center'], 5, (0, 255, 0), -1)
            cv2.putText(display, f"Lights:{len(lights)} Armors:{len(armors)} Thresh:{threshold_value}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        elif current_display == 4:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_red = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (180,255,255))
            display = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
        elif current_display == 5:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_blue = cv2.inRange(hsv, (100,100,100), (130,255,255))
            display = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow("Lights Test", display)
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q'): break
        elif key == ord('1'): current_display = 1
        elif key == ord('2'): current_display = 2
        elif key == ord('3'): current_display = 3
        elif key == ord('4'): current_display = 4
        elif key == ord('5'): current_display = 5
        elif key == ord('+') or key == ord('='): threshold_value = min(255, threshold_value+10)
        elif key == ord('-') or key == ord('_'): threshold_value = max(0, threshold_value-10)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
        if src.isdigit():
            src = int(src)
        test_lights_detection(src)
    else:
        test_lights_detection(0)