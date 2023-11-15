import cv2

cap=cv2.VideoCapture("video.mp4")

ret, img=cap.read()

det=cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=20)


count=0

while True:
    
    
    ret, img=cap.read()
    
    

    img = img[400: 1080, 320: 1400]
    
    mask = det.apply(img)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 2500:
            x,y,w,h= cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)
    
    
    cv2.imshow("mask", mask)
    cv2.imshow("frame", img)
    
    
    
    
    key=cv2.waitKey(13)
    if key==13:
        break
    
cv2.destroyAllWindows()
