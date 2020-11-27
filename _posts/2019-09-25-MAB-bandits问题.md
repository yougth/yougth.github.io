---
layout:     post
title:      MAB-bandits问题
subtitle:   E&E冷启动问题
date:       2019-09-30
author:     BY
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - bandits
    - Explore
    - Exploit
---

之前讲了[推荐系统重的EE算法](http://yougth.top/2018/12/25/ExploitationAndExploration/)，讲了一些基础的Bandits冷启动算法，参加Recsys发现论文中和工业界讲的占比最高的是MAB问题。这里接着上面在深入一下。

### LinUCB

前面说的这些MAB算法，做的假设是，当前这个新物料，是好的还是坏的是由本身的质量决定的。所以我们用点击率来定义物品质量，如果点击率高就会在exploit策略中大量推荐。

但是，现在的推荐系统已经是完全个性化的了，一个很小众的物料，比如小众游戏极品飞车，如果大量推荐，肯定点击率很低，但是如果只把它推荐给飞车迷们，就会点击率很高。所以这里隐含的假设是，物品的质量不仅和物品的本身质量有关，还和用户的一些特征相关。

所以我们不能用一个简单的点击率来表示，而是用一个向量来表示，向量的纬度是用户特征+物品特征。我们假设是d。对于每一个物品a，都学习1 x d的一组参数$$\theta_a$$，等到预估的时候，计算每一个物品的点击期望 $$ E[r_{t,a} \lvert x_{t,a}] = x_{t,a}^T \theta_a$$ ，这样就相当于回报和相关特征成线性关系，给MAB插上了特征的翅膀，经实验能够更快的收敛。上面公式中，其中t代表时间步，表示当前这一次推荐，另一个参数a表示表示物品a，$$x_{t,a}$$表示的在t时刻物品a的总特征向量，包括用户特征+物品特征。

从矩阵的角度出发，可以定义收益C

$$
D_{m \times d} \times \theta_{d \times 1} = C_{m \times 1}
$$

已知特征矩阵D和收益矩阵C，求解$$\theta$$

$$
\theta_{d \times 1} = (D_{m \times d}^T)^{-1} \times C_{x \times 1}
$$


对于单个物品a来说，假设收集到了m个feedback，假设特征矩阵为$$D_a$$，纬度m x d，假设我们收到的feedback为$$C_a$$，纬度为m x 1,用平方损失，定义损失函数为

$$
loss = (C_{x \times 1} - D_{m \times d} \theta_{d \times 1})^2 + \lambda \parallel \theta_{d \times 1} \parallel
$$

基本上和线性回归一样，不过后面加了一个正则化项，就是岭回归，为什么用岭回归？是因为MAB中一般样本数小于特征数，能够对回归参数进行修正，用最小二乘法求的最优解

$$
\theta_{d \times 1} = (D_{m \times d}^T \times D_{m \times d} + I_{d \times d})^{-1} \times D_{m \times d}^T C_{m \times 1}
$$

因为是UCB，所以需要一个置信上界

$$
a_t \approx arg \  max_{a \in A_t}(x_{t,a}^T \theta_a + \lambda \sqrt{x_{t,a}^T A_a^{-1} x_{t,a}})
$$

上面公式中x为$$d \times 1$$特征矩阵，其中

$$
A_{d \times d} \approx D_{m \times d}^T D_{m \times d} + I_{d \times d}
$$

这样

![算法流程](http://yougth.top/img/linUCB/linUCB.jpg)

首先第0行初始化，设置一个探索超参数 $$\alpha$$

1-11行代码是推荐的流程，目的是选择最优的可推荐物料。逐行解释的话，首先在时间步t，拿到所有arm的特征，或者在推荐中所说的待推荐物品的特征，对于每一个物品，如果是新物品，就用单位矩阵初始话$$A_a$$，用0向量初始化$$b_a$$，然后计算线性参数$$\theta$$，以及用线性参数和特征向量计算预估回报，同时加上置信区间结果，最终选择回报最高的做推荐。

11-14是更新参数的过程，首先拿到正是回报，更新$$A_{a,t}$$，更新$$b_{a,t}$$，算法结束。


算法的重点理解

 - 会考虑上下文特征，比如用户、物品、上下文特征，更容易收敛
 - 每一个物品维护自己的参数向量，各自更新，互不干扰
 - 和UCB一样，会用期望+置信上界，后半部分可以理解为潜力值，而且依然考虑了特征
 - 是在线学习算法，需要根据反馈结合实时特征进行更新物品的参数
 - 算法的缺点是，每一次推荐的计算复杂度为d x d x 物品数，所以首先向量纬度不能太大，另一个物品数不能太多，基本只能支持万级别以下的可推荐物品

我们可以看到，如果特征向量为1的话，LinUCB就退化成了基础的UCB。

后续学习完CLUB和COFIBA算法后接上更新


