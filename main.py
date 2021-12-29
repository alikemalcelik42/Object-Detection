import cv2
import trak

video = cv2.VideoCapture(r"C:\Users\alike\Videos\cars.mp4")
objectDetector = cv2.createBackgroundSubtractorMOG2(100, 50)

while video.isOpened():
    ret, frame = video.read()
    roi = frame[:170, :100]

    mask = objectDetector.apply(roi)
    mask = cv2.erode(mask, (1,1))
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (0, 255, 0), 2)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    cv2.imshow("Video", frame)

video.release()
cv2.destroyAllWindows()