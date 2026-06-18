import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Opened:", camera.isOpened())

ret, frame = camera.read()

print("Frame read:", ret)

if ret:
    print("Frame shape:", frame.shape)

camera.release()
