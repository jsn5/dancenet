import cv2
import numpy as np



VIDEO_PATH = 'data.mkv'
kernel = np.ones((2,2),np.uint8)
cap = cv2.VideoCapture(VIDEO_PATH)

data = []
count = 1
limit = 0
while(cap.isOpened()):
	ret, image_np = cap.read()
	if ret == False:
		break
	if limit == 3:
		limit = 0
		#image_np = 255 - image_np
		image_np = cv2.resize(image_np,(208,120))
		#ret,image_np = cv2.threshold(image_np,127,255,cv2.THRESH_BINARY)
		bg_index = np.where(np.greater(image_np,20))
		image_np[bg_index] = 255
		image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
		#(T, thresh) = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY)
		cv2.imwrite("imgs/{}.jpg".format(count),image_np)
		print("{}.jpg".format(count))
		count += 1
	limit += 1
