import cv2

cap=cv2.VideoCapture("video.mp4")

ret, img=cap.read()

det=cv2.createBackgroundSubtractorMOG2()


count=0

while True:
    
    
    ret, img=cap.read()
    
    

    img = img[400: 1080, 320: 1400]
    
    mask = det.apply(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 400:
            cv2.drawContours(img, [cnt], -1, (0,0,255))
        
    
    
    cv2.imshow("mask", mask)
    cv2.imshow("frame", img)
    
    
    
    
    key=cv2.waitKey(13)
    if key==13:
        break
    
cv2.destroyAllWindows()