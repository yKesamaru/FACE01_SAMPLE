import cv2
vcap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
while vcap.isOpened(): 
    ret, frame = vcap.read()
    if ret is True:
        cv2.imshow("TEST", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vcap.capture.release()
    cv2.destroyAllWindows()