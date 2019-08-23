import numpy as np 
import cv2

"""仿射矫正
就是从图像上找到vanishing line，把它映射到规范位置 (0,0,1)^T
"""



"""通过一组场景平行线获得vanishing point
paras:
	parallels [num x 3] (homogeneous form)
return:
	vanishing_point [1 x 3]
"""
def get_vanishing_point(parallels,mode='center'):
	"""
	算法思路：分别计算2条线的交点
	if mode == center: 取其中一个点，使得它到其他点的距离最小 (抗噪性强)
	if mode == mean: 取均值
	"""

	# step1 去掉图像平行线
	temp1 = np.repeat(parallels,parallels.shape[0],axis=0).reshape((parallels.shape[0],-1,3))
	temp2 = np.tile(parallels,(parallels.shape[0],1,1))
	res = np.cross(temp1,temp2)

	keep = []
	flag = []
	for index,item in enumerate(res):
		if index in flag:
			continue
		check = np.where((item[:,0] == 0)&(item[:,1] == 0) & (item[:,2] == 0 ))[0]
		if len(check) == 1:
			keep.append(check[0])
		elif len(check) > 1:
			keep.append(check[0])
			flag.extend(check[np.where(check != index)])

	res = res[keep]
	res = res[:,keep,:]

	# step2 计算vanishing point
	point_list = []
	for index,item in enumerate(res[:-1]):
		for point in item[index+1:]:
			point = point/point[2]
			point_list.append(point[:2])
	
	point_list = np.array(point_list)

	if mode == 'mean':
		vanishing_point = np.mean(point_list,axis=0)
	elif mode == 'center':
		temp1 = np.repeat(point_list,point_list.shape[0],axis=0).reshape((point_list.shape[0],-1,2))
		temp2 = np.tile(point_list,(point_list.shape[0],1,1))
		dis = np.mean(np.linalg.norm(temp1 - temp2,axis=2),axis=1)
		vanishing_point = point_list[np.argmin(dis)]

	vanishing_point = np.append(vanishing_point,1) # homogeneous 

	return vanishing_point


"""矫正
通过把消影线规范到(0,0,1)^T
pars:
	img: 输入图片
	parallel_A: 平行线组1 [num x 4] [x1,y1,x2,y2]
	parallel_B: 平行线组2 [num x 4] [x1,y1,x2,y2]
return:
	矫正的图片
"""
def rectify(img, parallel_A, parallel_B, H_A=None, debug=False):

	# lines [num x 4]
	def homogeneous_form(lines):
		return np.cross(np.hstack([lines[:,:2],np.ones((lines.shape[0],1))]),
			            np.hstack([lines[:,2:],np.ones((lines.shape[0],1))]))

	homogeneous_parallel_A = homogeneous_form(parallel_A)
	homogeneous_parallel_B = homogeneous_form(parallel_B)

	vanishing_point_A = get_vanishing_point(homogeneous_parallel_A)
	vanishing_point_B = get_vanishing_point(homogeneous_parallel_B)

	vanishing_line = np.cross(vanishing_point_A,vanishing_point_B)
	vanishing_line /= np.sum(vanishing_line)

	if debug is True:
		min_x = min(vanishing_point_A[0],vanishing_point_B[0])
		max_x = max(vanishing_point_A[0],vanishing_point_B[0])
		min_y = min(vanishing_point_A[1],vanishing_point_B[1])
		max_y = max(vanishing_point_A[1],vanishing_point_B[1])

		new_width = int(max(max_x,img.shape[1]) - min(min_x,0) + 200)
		new_height = int(max(max_y,img.shape[0]) - min(min_y,0) + 200)
		new_img = np.ones((new_height,new_width,3)) * 255

		new_origin = np.array([min(min_x,0) - 50, min(min_y,0) - 50]).astype(np.int)

		offset_x = 0 - new_origin[0]
		offset_y = 0 - new_origin[1]

		new_vanishing_point_A = vanishing_point_A + [offset_x,offset_y,0]
		new_vanishing_point_B = vanishing_point_B + [offset_x,offset_y,0]

		new_img[offset_y:offset_y+img.shape[0],offset_x:offset_x+img.shape[1]] = img.copy()

		new_parallel_A = parallel_A + [offset_x,offset_y,offset_x,offset_y]
		new_parallel_B = parallel_B + [offset_x,offset_y,offset_x,offset_y]

		for line in new_parallel_A:
			cv2.line(new_img,(line[0],line[1]),(line[2],line[3]),(255,0,0),3)
			cv2.line(new_img,(line[0],line[1]),(int(new_vanishing_point_A[0]),int(new_vanishing_point_A[1])),(255,0,0),3)

		for line in new_parallel_B:
			cv2.line(new_img,(line[0],line[1]),(line[2],line[3]),(0,0,255),3)
			cv2.line(new_img,(line[0],line[1]),(int(new_vanishing_point_B[0]),int(new_vanishing_point_B[1])),(0,0,255),3)

		cv2.line(new_img,(int(new_vanishing_point_A[0]),int(new_vanishing_point_A[1])),
			(int(new_vanishing_point_B[0]),int(new_vanishing_point_B[1])),(0,0,0),3,cv2.LINE_AA)

		cv2.circle(new_img,(int(new_vanishing_point_A[0]),int(new_vanishing_point_A[1])),5,(0,0,0),10)
		cv2.circle(new_img,(int(new_vanishing_point_B[0]),int(new_vanishing_point_B[1])),5,(0,0,0),10)

		cv2.imwrite('description.png',new_img)

	length = max(img.shape)

	if H_A is None:
		a11 = 1; a12 = 1; a21 = 0; a22 = 1
		tx = 0; ty = length/3
		H_A = np.array([
			[a11,a12,tx],
			[a21,a22,ty],
			[0,0,1]
			]).astype(np.float64)

	H_p = np.array([
		[1,0,0],
		[0,1,0],
		[vanishing_line[0],vanishing_line[1],vanishing_line[2]]
		])

	H = np.matmul(H_A,H_p)
	img = cv2.warpPerspective(img,H,(length,length))


	print('before vanishing_line:',vanishing_line)
	try:
		vanishing_line = np.matmul(np.linalg.inv(H).transpose((1,0)),vanishing_line)
	except:
		vanishing_line = np.matmul(np.linalg.pinv(H).transpose((1,0)),vanishing_line)
	print('after vanishing_line:',vanishing_line)

	return img


if __name__ == '__main__':
	import sys
	img = cv2.imread(sys.argv[1])

	linesA = np.array([[304,421,558,181],[108,251,430,55]])
	linesB = np.array([[306,420,42,191],[412,320,84,98],[555,181,296,56]])

	out = rectify(img,linesA,linesB,debug=True)
	cv2.imwrite('res.png',out)
