import numpy as np
from algorithm import compute_homography
import cv2,sys


# point_list1 = np.array([[212,186,1],[300,134,1],[387,183,1],[301,248,1]])

# point_list2 = np.array([[206,308,1],[303,213,1],[398,308,1],[302,404,1]])


# h1 = dlt.basic_dlt(point_list1,point_list2)
# print(h1)
# print(np.linalg.norm(h1))


# h2 = dlt.normilized_dlt(point_list1,point_list2)
# print(h2)
# print(np.linalg.norm(h2))


# h,_ = cv2.findHomography(point_list1[:,:2],point_list2[:,:2])
# print(h)
# print(np.linalg.norm(h))


# pic = cv2.imread(sys.argv[1])

# length = max(pic.shape)
# img = cv2.warpPerspective(pic,h2,(length,length))
# cv2.imwrite('res.png',img)


pic = cv2.imread(sys.argv[1])
cv2.imshow('pic',pic)

cv2.waitKey(0)