import numpy as np



"""Basic DLT algorithm
para:
	point_list1: [X1,X2,X3,X4,...Xn]
	for each X: [x1,y1,w1]
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

	u,s,vt = np.linalg.svd(A,full_matrices=0,compute_uv=1) # 这里 s 从大到小排列
	h = vt[-1].reshape((3,3))
	
	return h




		

