import numpy as np
from dlt_algorithm import dlt

point_list1 = np.array([[141,131,1],[480,159,1],[493,630,1],[64,601,1]])

point_list2 = np.array([[318,256,1],[534,372,1],[316,670,1],[73,473,1]])

h = dlt.basic_dlt(point_list1,point_list2)
print(h)
print(np.linalg.norm(h))