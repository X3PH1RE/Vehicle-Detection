import cv2




import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids










tracker = EuclideanDistTracker()


cap=cv2.VideoCapture("video.mp4")

ret, img=cap.read()

det=cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=20)


count=0

while True:
    
    
    ret, img=cap.read()
    
    

    img = img[400: 1080, 320: 1400]
    
    mask = det.apply(img)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 2500:
            x,y,w,h= cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)
            detections.append([x,y,w,h])

        print(detections)
    
    
    boxes_id = tracker.update(detections)
    for id in boxes_id:
        x, y, w, h, id = id
        cv2.putText(img, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)

    
    cv2.imshow("mask", mask)
    cv2.imshow("frame", img)
    
    
    
    
    key=cv2.waitKey(50)
    if key==27:
        break
    
cv2.destroyAllWindows()