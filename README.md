# Estimation - 2D Projective Transformations

[TOC]

## 1.The Direct Linear Transformation (DLT) algorithm

**1.1.The Gold Standard algorithm.** The computational algorithm that enables this cost function to be minimized is called the "Gold Standard" algorithm.

**1.2.The Direct Linear Transformation(DLT) algorithm**

由于是齐次变量，$x'_i$和$Hx_i$可能不相等（差一个scale）。但是方向相同，所以叉乘为0。即：
$$
x'_i \times Hx_i = 0
$$
设${\bf x'_i}   = (x'_i,y'_i,w'_i)^T$ , 最后可得如下：
$$
\left[
\begin{matrix}
{\bf 0}^T&-w'_i{\bf x}_i^T&y'_i{\bf x}_i^T\\w'_i{\bf x}_i^T&{\bf 0}^T&-x'_i{\bf x}_i^T
\end{matrix}
\right]
\left(
\begin{matrix}
{\bf h}^1\\{\bf h}^2\\{\bf h}^3
\end{matrix}
\right) = {\bf 0}
$$
简化为：
$$
{\bf A}_i{\bf h} = {\bf 0}
$$
其中$A_i$ 是一个$2 \times 9$ 的矩阵。

**Over-determined solution** 首先为了避免$\bf h = 0$ ,可设定$||{\bf h}|| = 1 $ 。

超定情况下，如下解决：
$$
minimize \frac {||{\bf Ah}||} {||{\bf h}||}  \ \ \ s.b.t \ \ \ \ \ ||\bf h|| = 1
$$

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="pics/8.png" height="200">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 1px;">basic DLT for H</div>
</center>

*约束数量大于自由度是解H的条件。但是混合类型下（如提供的约束包含点也包含线）不一定，如2点2线不可以唯一解出H，但是3点1线或者1线3点可以（in general position）。*

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="pics/9.png" height="80">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 1px;">A configuration of two points and two lines is equivalent to five lines with four concurrent, or five points with four collinear</div>
</center>

##  2.Different cost functions

**2.1. 代数距离 (Algebraic distance)**
$$
d_{alg}({\bf x}_1,{\bf x}_2)^2 = a_1^2 + a_2^2 \ \ where \ {\bf a} = (a_1,a_2,a_3)^T = x_1 \times x_2
$$
**2.2. 几何距离(Geometric distance)**

*说明 x表示测量值，$\hat x $表示估计值, $\bar x$ 表示真值。*

1. **单图误差**(误差只存在于第二张图)
   $$
   \sum_i d({\bf x}'_i,H \bar {\bf x}_i)^2
   $$
   其中d表示欧式距离。

   

2. **对称转移误差(Symmetric transfer error)**

   <center>
   <img style="border-radius: 0.3125em;
   box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
   src="pics/10.png" height="100">
   <br>
   <div style="color:orange; border-bottom: 1px solid #d9d9d9;
   display: inline-block;
   color: #999;
   padding: 1px;">对称转移</div>
   </center>

   $$
   \sum_id({\bf x}_i,H^{-1}{\bf x}'_i)^2 + d({\bf x}'_i,H{\bf x}_i)^2
   $$

   

3. **重映射误差---(双图)(Reprojection error)**

   <center>
   <img style="border-radius: 0.3125em;
   box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
   src="pics/11.png" height="100">
   <br>
   <div style="color:orange; border-bottom: 1px solid #d9d9d9;
   display: inline-block;
   color: #999;
   padding: 1px;">重映射</div>
   </center>

   对于测量值$x$，寻找一个homography $\hat H$ 和 一对完全匹配的点 $\hat x_i$和$\hat x'_i$ ,并最小化以下误差：	
   $$
   \sum_id({\bf x}_i.\hat {\bf x}_i)^2 + d({\bf x}'_i,\hat {\bf x}'_i)^2 \ \ subject \ to \ \hat{\bf x}'_i = \hat H \hat {\bf x}_i \ \forall i
   $$
   *仿射变换下，geometric distance 和 algebraic distance 是相同的*

   **Reprojection error的几何理解**

   两个平面之间homography的估计可以被认为是在4D空间用**面**来拟合**点**。

   因此给定一个4D空间点：
   $$
   {\bf X}_i = (x_i,y_i,x'_i,y'_i)^T
   $$
   估计homography的任务变成了找到一个穿过点${\bf X}_i$参数簇${\cal  V}_H$。由于${\cal V}_H$定义的超曲面不太可能完
   
   美穿过${\bf X}_i$，因此，让${\cal V}_H$ 上的点${\bf \hat X}_i = (\hat x_i,\hat y_i,\hat x'_i,\hat y'_i)^T$ 成为距离${\bf X}_i$最近的点:
   $$
   ||{\bf X}_i - \hat {\bf X}_i||^2 = (x_i - \hat x_i)^2 + (y_i - \hat y_i)^2 + (x'_i -\hat x'_i)^2 + (y'_i - \hat y'_i)^2\\ = d({\bf x}_i,\hat {\bf x}_i)^2 + d({\bf x}'_i,\hat {\bf x}'_i)^2
   $$
   由此可见这等价于求**reprojection error**。
   
   ${\bf X}$ 和 ${\bf \hat X}$ 之间的最短距离垂直于平面${\cal V}_H$ 的切平面。因此：
   $$
   d({\bf x}_i,\hat {\bf x}_i)^2 + d({\bf x}'_i,\hat {\bf x}'_i)^2 = d_{\bot}({\bf X}_i,{\cal V}_H)^2
   $$

4. **Sampson 误差(Sampson error)** 

   Sampson error的思想是估计$\hat {\bf X}$的<font color=blue>一阶渐进</font>。

   给定一个homography H，任何点${\bf X} = (x,y,x',y')^T$ 若位于${\cal V}_H$上则满足$\bf Ah = 0$ ，这里表示为$\bf {\cal C}_H(X) = 0$ ,其中 $\bf {\cal C}_H$ 是一个2-vector。 

   根据 Taylor expansion 可以得到：
   $$
   \bf {\cal C}_H(X + \delta_X) = {\cal C}_H(X) + \frac {\partial{\cal C}_H} {\partial X} \delta_X\\
   where \ \delta_X = \hat X - X \ and \ \hat X \ lie \ on \ {\cal V}_H \to {\cal C}_H(\hat X)= 0\\
   \therefore {\cal C}_H(X) + \frac {\partial{\cal C}_H} {\partial X} \delta_X = 0 \Longrightarrow \epsilon + J \delta_X = 0\\
   where\ \epsilon = {\cal C}_H(X) \ and \  J = \frac {\partial{\cal C}_H} {\partial X} \\
   \therefore J \delta_X = - \epsilon
   $$
   目标是最小化$\bf X$与$\bf \hat X$的距离，即$\bf \delta_x = \hat X - X$ 。所以当前问题变为：
   $$
   \bf \text{find vector } \bf \delta_X \text {to minimize }||\bf \delta _X|| \text { subject to } \bf J \delta_X = - \epsilon。
   $$
   这里使用Larange multiplers来求解：
   $$
   \bf F = \delta_X^T \delta_X - 2\lambda^T(J\delta_X+\epsilon) \ 的极值\\
   \frac {\partial F} {\partial \delta_X} = 2\delta_X^T - 2\lambda^TJ = 0^T \Rightarrow \delta_X = J^T\lambda\\
   \because J\delta_X + \epsilon = 0 \ 代入\delta_X可得\\
   JJ^T\lambda = - \epsilon  \Rightarrow \lambda = -(JJ^T)^{-1}\epsilon\\
   \therefore \delta_X = J^T\lambda = -J^T(JJ^T)^{-1}\epsilon\\
   The \ norm \ ||\delta_X||^2 \ is \ Sampson \ error:\\
   ||\delta_X||^2 = \delta_X^T\delta_X = \epsilon^T(JJ^T)^{-1}\epsilon
   $$
   

## 3.Statistical cost functions and Maximum Likelihood estimation

设想$x = \bar x + \Delta x$ ,其中 $\Delta x$ 服从Gaussian分布(variance = $\sigma^2$ ) 。

首先高斯分布：
$$
f(x) = \frac {1} {\sqrt {2\pi} \sigma}exp(- \frac {(x - \mu)^2} {2 \sigma^2})
$$
如何真值点是$\bf \bar x$对于每一个测量点x的Probability density function(PDF)：	
$$
\bf Pr(x) = (\frac {1} {2\pi \sigma^2})e^{-d(x,\bar x)^2/(2 \sigma^2)}
$$

1. **单图误差(error in one image)**

   假设误差只在第二张图存在，视每个点的误差$\bf \Delta x$是独立同分布的(Gaussian)，所以可得：
   $$
   \bf Pr({\{x'_i\}|H}) = \prod_i (\frac {1} {2\pi \sigma^2})e^{-d(x'_i,H\bar x_i)^2/(2 \sigma^2)}
   $$
   <font color='red'>这里</font>$\color {red} {\bf Pr({x'_i}|H)}$<font color='red'>可以被理解为在被给定真实homography H后，获得测量值</font>$\color {red} {\bf \{x'_i\}}$<font color='red'>的概率。</font>

   对上式取$log-likelihood$ 可以得到:
   $$
   \bf log Pr({\{x'_i\}|H}) = - \frac {1} {2 \sigma^2} \sum_i d(x'_i,H \bar x_i)^2 + constant 
   $$
   对$\hat H$使用最大似然估计(MLE)，最大化log等价于最小化:
   $$
   \bf \sum_i d(x'_i,H \bar x_i)^2
   $$
   *可以看到这里MLE 等价于最小化几何距离*

2. **双图误差(error in both images)**
   $$
   \bf Pr(\{x_i,x'_i\}|H,\{\bar x_i \}) = \prod_i(\frac {1} {2\pi \sigma^2})^2e^{-(d(x_i,\bar x_i)^2 + d(x'_i,H \bar x_i)^2)/(2 \sigma^2)}
   $$
   MLE对H和$\bf \{x_i \leftrightarrow x'_i\}$ 的估计，即$\hat H$ 和$\bf \{\hat x_i \leftrightarrow \hat x'_i\}$，即最小化:
   $$
   \bf \sum_i d(x_i,\hat x_i)^2  + d(x'_i,\hat x'_i)^2\\
   with \ \hat x'_i = \hat H \hat x_i
   $$
   可以看出这里MLE与重映射误差是等价的。

3. **马氏距离(Mahalanobis distance)**

   更加一般化的高斯情况下，可以设想测量值X满足一个带有协方差矩阵的高斯分布。最大化log-likelihood等价于最小化马氏距离:
   $$
   \bf ||X - \bar X||^2_\Sigma = (X-\bar X)^T \Sigma^{-1}(X-\bar X)
   $$
   两张图片的的误差是独立的，因此最后的损失函数为:
   $$
   \bf ||X - \bar X||^2_\Sigma + ||X' - \bar X'||^2_{\Sigma'}
   $$

## 4.Transformation invariance and normalization

1. **图片内部坐标转换下的不变量 Invariance**

   （i) 图片坐标变换$\bf \tilde x = Tx$ 和 $\bf \tilde x' = T'x'$ 。

     (ii) 找到新的holography $\bf \tilde H$对应于$\bf \tilde x_i \leftrightarrow \tilde x'_i$。

     (iii) 设定 $\bf H = T'^{-1} \tilde H T$

   <font color='red'>需要关注的问题是算法的结果是否独立于</font>$\bf \color {red} {T,T'}$ 

   *显然在相似变化下最小化几何距离的算法是不变的，但是上述的DLT算法在相似变换下结果会变。*

   

2. **DLT算法下的Non-invariance**

   **定理** 设$T'$是一个相似变换(scale 是s)；$T$是一个任意的射影变换；$H$是一个2D单应并且$\tilde H = T'HT^{-1}$。那么可以得到$||\tilde A \tilde h|| = s||Ah||$。
  
  可以用代数距离algebraic distance来表示：
  $$
  \bf d_{alg}(\tilde x'_i,\tilde H \tilde x_i) = s d_{alg}(x'_i,Hx_i)
  $$
  $\bf H$和$\bf \tilde H$之间不存在给出同样的误差$\epsilon$ 并同时满足$\bf ||H|| = ||\tilde H||=1$的意义对应，具体地说,
  $$
  \bf
  minimize\ \sum_i d_{alg}(x'_i,Hx_i)^2 \text { subject to } ||H||=1 \\
  \iff minimize\ \sum_i d_{alg}(\tilde x'_i,\tilde H \tilde x_i)^2 \text{ subject to } ||H|| = 1 \qquad \quad \\
  \nLeftrightarrow minimize\ \sum_i d_{alg}(\tilde x'_i,\tilde H \tilde x_i)^2 \text{ subject to } ||\tilde H|| = 1 \quad \;
  $$

3. **几何误差下的invariance**
   $$
   \bf d(\tilde x',\tilde H \tilde x) = d(T'x',T'HT^{-1}Tx) = d(T'x',T'Hx) = d(x',Hx)
   $$
   最后一项等式成立是因为欧式距离在欧式变换下不变。在相似变换下几何误差只相差一个scale，因此最小化几何误差在相似变换下是不变的。

   

4. **Normalizing transformation**

   Normalization包括坐标的translation和scaling。其优点为：

   - 提高结果的准确度
   - 使用初始数据normalization的算法对于坐标原点和尺寸的任意选择是invariant的。因此DLT算法可以在相似变换下invariant。

   **Isotropic scaling** :

   + normalization的第一步是把所有点的形心移动到坐标原点。
   + 对所有点的坐标尺寸进行缩放，使得对于坐标原点的平均距离为$\sqrt 2$。
   + 把这个变换独立应用于两张图片。

   <center>
   <img style="border-radius: 0.3125em;
   box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
   src="pics/12.png" height="200">
   <br>
   <div style="color:orange; border-bottom: 1px solid #d9d9d9;
   display: inline-block;
   color: #999;
   padding: 1px;">normalized DLT for H</div>
   </center>

   **Non-isotropic scaling**：

   + 同样，第一步把所有点的形心移动到坐标原点。
   + 进行坐标尺度缩放，使得该点集的两主距都等于1。这样形成了以原点为中心，半径为1的近似的对称圆云。

## 5.Iterative minimization methods

迭代最小化技术5个步骤：

+ 损失函数：需要提供cost function来最小化
+ 参数：需要计算的transformation需要用有限的参数来表示，通过参数选择可以限制变换的类型
+ 明确的函数：需要明确的函数来表达cost
+ 初始化：需要合适的初始参数估计
+ 迭代：从初始化的解开始，参数通过最小化损失函数被迭代优化

**明确函数**:

1. **单图误差**
   $$
   \bf \text {Minimize } \sum_i d(x'_i,H\bar x_i)^2
   $$
   

   首先测量值$\bf X$由点$\bf x'_i$的$2n$个非齐次坐标组成。把homograohy的实体$h$当作参数：
   $$
   f:\bf h \to (Hx_1,Hx_2,...,Hx_n)
   $$
   

   只需最小化9个参数。

2. 对称转移误差 Symmetric transfer error**
   $$
   \bf \text {Minimize } \sum_id(x_i,H^{-1}x'_i)^2+d(x'_i,Hx_i)^2
   $$
   测量值$\bf X$是由点$\bf x_i$和点$\bf x'_i$的$4n$个非齐次坐标组成。把homograohy的实体$h$当作参数：
   $$
   f:\bf h \to (H^{-1}x'_1,..,H^{-1}x'_n,Hx_1,...,Hx_n)
   $$
   

   只需最小化9个参数。

3. **重映射误差 Reprojection error**

   该问题需要参数为$2n+9$ ,包括 ($\bf \hat x_i,\hat x'_i,\hat H$)。因此参数$\bf P = (h,\hat x_1,\hat x_2,…,\hat x_n)$

   函数$f$定义为:
   $$
   f:\bf (h,\hat x_1,...,\hat x_n) \to (\hat x_1,\hat x'_1,...,\hat x_n,\hat x'_n)
   $$
   测量值$\bf X$是4n-vector,即$\bf X_i = (x_i,y_i,x'_i,y'_i)^T$。

4. **Sampson 近似**

   Sampson近似可以使得重映射误差只需最小化9个参数。对于每一个确定的h，Sampson error确定了$2n$个变量$\bf \{\hat x_i\}$。

**初始化**

推荐两种初始化技术：1.线性技术 Linear techniques  2.最小化方法 minimal solutions

初始化2种方法：

1. 每次从数据中作满足参数空间的采样，迭代选择采样点(可以随机选或者按照某种方式选)并保留最好的结果;(这种方法仅在参数空间维度较小时适用)
2. 从参数空间的固定点初始化。(通常不可行，迭代会陷入错误最小值或者不收敛)

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="pics/13.png" height="400">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 1px;">The Gold Standard algorithm and variations for estimating H from image correspon- dences. The Gold Standard algorithm is preferred to the Sampson method for 2D homography compu- tation.</div>
</center>

**迭代方法**

1. Newton iteration
2. Levenberg-Marquardt method

## 6.Robust estimation

**1.RANSAC  (RANdom Sample Consensus)算法**

问题描述：在一个2D点集合上寻找一条直线，最小化垂直平方距离，并约束于任何无用点距离这条线至少t个单元。

思想：随机从2D点集合选择2个点，这两个点定义了一条直线。这条直线的support指的是在阈值距离内的点数。随机选择需要重复进行几次，拥有最多support的直线被认为是最好的拟和这些点。

在阈值距离内的点称为inliers；在阈值距离外的点称为outliers。

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="pics/14.png" height="300">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 1px;">RANSAC算法描述</div>
</center>

几个问题：

1. **距离阈值是多少？**

   在实际当中通常通过经验选择。

   假设测量误差是均值为0，标准差为$\sigma$的高斯分布。在这个情况下,$d^2_{\bot}$是高斯变量的平方和，服从$\chi^2_{m}$分布，其中$m$代表自由度。

   $\chi^2_m$随机变量小于$k^2$的概率是$F_m(k^2) = \int ^{k}_0 \chi^2_m(\xi)d\xi$ 。通过下面来判断inlier和outlier：
   $$
   \begin{cases}
   inlier&d^2_\bot < t^2\\
   outlier&d^2_\bot \ge t^2
   \end{cases}
   \ with \ t^2 = F^{-1}_m(\alpha)\sigma^2 
   $$
   通常$\alpha$取0.95。

2. **取多少次样本？**

   取样次数需要足够高来保证一个概率$p$，这个概率是s个点组成的随机样本中至少有一次没有outliers的概率。通常p的取值为0.99。

   *至少一次即$\ge 1$，所以1-p可以理解为0次没有outlier的概率，即都有outlier的概率*

   假设$w$为选择的点是inlier的概率，$\epsilon = 1 - w$是outlier的概率。然后至少需要$N$次选择，其中$(1-w^s)^N = 1 - p$，因此：
   $$
   N = log(1-p)/log(1-(1-\epsilon)^s)
   $$

3. **可以接收到consensus集合是多大？**

   对于n个数据点，$T = (1- \epsilon)n$

**自适应决定N的大小（算法4.5）**

+ N = $\infty$, sample_count = 0

+ 当 N > sample_count 时，重复

  — 选择一个样本，并计算inliers的数目

  — 求 $\epsilon = 1 - (\text{number of inliers})/(\text{total number of points})$

  — 求 $N = log(1-p)/log(1-(1-\epsilon)^s)$，其中p = 0.99

  — sample_count += 1

+ 结束

## 7.Automatic computation of a homography

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="pics/15.png" height="400">
<br>
</center>

**确定假设的conrrespondences**

目的是在不知道单应的情况下提供初始的点对应集合（会存在误差）。这些假设由在每一幅图像独立地检测兴趣点获得，然后利用领域灰度的近似度和相似度的组合来匹配这些兴趣点。可称兴趣点为角点，角点定义为图像自相关函数的最小值。

一幅图像上的一个corner可能被另一幅上不止一个角点所"认定"。在这种情况下，采取winner takes all策略，即选择最高互相关的匹配。

相似测量的另一种方法是用灰度差的平方和（SSD Squard Sum of intensity Differences）来代替互相关（CC Cross-Correlation）。图像之间的灰度值变化不大时，通常倾向于用SSD，因为它的测量比CC灵敏且计算代价较小。

**用RANSAC测量homography**

样本大小 s = 4,因为4组点对应确定一个homography。采样次数自适应调整。

1. **距离测量**:显然有3种方式

   (i) 对称转移误差:
   $$
   d^2_{transfer} = d(x,H^{-1}x')^2 + d(x',Hx)^2
   $$
   (ii) 重映射误差：
   $$
   d^2_{\bot} = d(x,\hat x)^2 + d(x',\hat x')^2\\
   where \quad \hat x' = H \hat x \quad \text{is the perfect correspondence}
   $$
   (iii)Sampson误差

2. **样本选择**:这里有2个问题

   (i)退化的样本需要被丢弃，比如4点中有3点共线

   (ii) 组成样本的点在整个图像上有合理的空间分布

**鲁棒ML估计与引导匹配**

双重目标：

1. 用所有估计得到的inliers来估计homography；
2. 从假设对应集获得更多的 inlying matches 因为得到了一个更加准确的homography 。

方法：

在inliers上执行一个ML估计，然后用新估计的homography重新计算inliers，重复执行直到inliers的数目收敛。