import numpy as np
from algorithm import dlt
import cv2


point_list1 = np.array([[141,131,1],[480,159,1],[493,630,1],[64,601,1]])

point_list2 = np.array([[318,256,1],[534,372,1],[316,670,1],[73,473,1]])

# print('centroid:',np.mean(point_list1,axis=0))
# print('mean_distance:',np.mean(np.linalg.norm(point_list1[:,:2],axis=1)))
h1 = dlt.basic_dlt(point_list1,point_list2)
print(h1)
print(np.linalg.norm(h1))


h2 = dlt.normilized_dlt(point_list1,point_list2,False)
print(h2)
print(np.linalg.norm(h2))


h,_ = cv2.findHomography(point_list1[:,:2],point_list2[:,:2])
print(h)
print(np.linalg.norm(h))