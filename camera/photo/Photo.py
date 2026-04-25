import cv2
camera=cv2.VideoCapture(0)
i=0
print("press s save,press q quit")
while True:
    ret,img=camera.read()
    if not ret:
        break
    cv2.imshow('Camera',img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('s'):
        i+=1
        filename=f'catch_img_{i}.jpg'
        cv2.imwrite(filename,img)
        print(f'save:{filename}')
    elif key==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()