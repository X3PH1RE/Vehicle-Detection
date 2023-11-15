import cv2

cap=cv2.VideoCapture("video.mp4")

ret, img=cap.read()
row,col,_=img.shape
while True:
    
    
    #selecting the region of interest
    ret, img=cap.read()
    
    img = img[350: row, 170: col]
    
    cv2.imshow("roi", img)
    key=cv2.waitKey(13)
    if key==13:
        break
    
cv2.destroyAllWindows()