import numpy as np
from algorithm import dlt
import cv2,sys


point_list1 = np.array([[10,100,1],[281,37,1],[499,341,1],[93,492,1]])

point_list2 = np.array([[62,55,1],[424,54,1],[448,564,1],[53,573,1]])


h1 = dlt.basic_dlt(point_list1,point_list2)
print(h1)
print(np.linalg.norm(h1))


h2 = dlt.normilized_dlt(point_list1,point_list2)
print(h2)
print(np.linalg.norm(h2))


h,_ = cv2.findHomography(point_list1[:,:2],point_list2[:,:2])
print(h)
print(np.linalg.norm(h))


pic = cv2.imread(sys.argv[1])

img = cv2.warpPerspective(pic,h2,(pic.shape[1],pic.shape[0]))
cv2.imwrite('res.png',img)
# pic = cv2.imread(sys.argv[1])
# cv2.imshow('pic',pic)

# cv2.waitKey(0)