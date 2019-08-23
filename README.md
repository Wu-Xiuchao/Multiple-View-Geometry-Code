# Multiple View Geometry Code 

Code about MVG

## Compute Homography

```python
from algorithm import compute_homography
```

```
basic_dlt(point_list1,point_list2)
normilized_dlt(point_list1,point_list2)
```

<table border=0 cellspacing=0 cellpadding=0>
<tr><td><img src="pngfiles/projgeomfigs-floor.persp.bmp"/></td><td><img src="pngfiles/projgeomfigs-floor.fronto.bmp"/></td>
  <td><img src="pngfiles/homography_res.png"></td>
  </tr>
  <tr><td><div align="center">persp</div></td>
    <td><div align="center">fronto</div></td>
    <td><div align="center">result</div></td>
  </tr>
</table>

## Affine Rectification

Transform the vanishing line to its canonical form ![](http://latex.codecogs.com/gif.latex?\ (0,0,1)^T)

```python
from algorithm import affine_rectification
```

```
rectify(img, parallel_A, parallel_B, H_A=None, debug=False)
```

<table border=0 cellspacing=0 cellpadding=0>
<tr><td><img src="pngfiles/description.png"/></td><td><img src="pngfiles/affine_rectify_res.png"/></td>
  </tr>
  <tr><td><div align="center">vanishing line</div></td>
    <td><div align="center">rectified</div></td>
  </tr>
</table>

