import cv2

video = cv2.VideoCapture('./results/medical128.gif')
success, frame = video.read()
a = []
while(success):
    a = frame
    success, frame = video.read()
#cv2.imwrite('a.jpg',a)