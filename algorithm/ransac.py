import numpy as np 
import cv2

"""RANSAC
paras:
	data: input data S [num x dimension]
	
return:
	subset [sub_num x dimension]
	model [1 x dimension]
"""
def ransac(data,t,T):

	s = data.shape[1]

	subset = data[np.random.randint(data.shape[0],size=s),:] # 选择初始数据集 [s x s]

	u,s,vt = np.linalg.svd(subset,full_matrices=0,compute_uv=1)
	model = vt[-1].reshape((1,s))

	return subset,model


if __name__ == '__main__':
