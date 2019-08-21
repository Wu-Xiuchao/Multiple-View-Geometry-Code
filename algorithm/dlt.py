import numpy as np
import math


"""Basic DLT algorithm
para:
	point_list1: [X1,X2,X3,X4,...Xn] (point_list1 * H = point_list2)
	for each X: [x1,y1,w1]
return:
	homography
"""
def basic_dlt(point_list1,point_list2):
	A = []
	for point1,point2 in zip(point_list1,point_list2):
		Ai = np.zeros((2,9))
		Ai[0][3:6] = -point2[2] * point1
		Ai[0][6:9] = point2[1] * point1 
		Ai[1][0:3] = point2[2] * point1
		Ai[1][6:9] = -point2[0] * point1
		if len(A) == 0:
			A = Ai 
		else:
			A = np.vstack([A,Ai])

	u,s,vt = np.linalg.svd(A,full_matrices=1,compute_uv=1) # 这里 s 从大到小排列

	h = vt[-1].reshape((3,3))
	
	return h

"""Normilized DLT algorithm
para:
	point_list1: [X1,X2,X3,X4,...Xn] (point_list1 * H = point_list2)
	for each X: [x1,y1,w1]
return:
	homography
"""
def normilized_dlt(point_list1,point_list2):

	def normilize(data):
		s = math.sqrt(2) * (1/np.mean(np.linalg.norm(data[:,:2]-np.mean(data[:,:2],axis=0),axis=1)))
		t = -s * np.mean(data,axis=0)
		theta = 0
		T = np.array([[s*math.cos(theta),-s*math.sin(theta),t[0]],
			          [s*math.sin(theta),s*math.cos(theta),t[1]],
			          [0,0,1]])
		data = np.matmul(T,data.transpose((1,0))).transpose((1,0))
		return data,T

	point_list1,T = normilize(point_list1)
	point_list2,T_ = normilize(point_list2)

	tilde_h = basic_dlt(point_list1,point_list2)

	h = np.matmul(np.matmul(np.linalg.pinv(T_),tilde_h),T)

	return h


	


		

