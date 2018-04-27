Google机器学习开源课程笔记

[TOC]

## 机器学习概念

### 1.框架处理

#### 常见术语

- 标签：预测的事物，y变量
- 特征：输入变量，x变量
- 模型：定义了x与y之间的关系

监督学习：

- 分类：离散值（好坏）
- 回归：连续值（价格）

无监督学习：

- 聚类




### 2.深入了解机器学习

#### 训练与损失：

##### 损失 (Loss)

一种衡量指标，用于衡量模型的[**预测**](https://developers.google.cn/machine-learning/crash-course/glossary#prediction)偏离其[**标签**](https://developers.google.cn/machine-learning/crash-course/glossary#label)的程度。或者更悲观地说是衡量模型有多差。要确定此值，模型必须定义损失函数。例如，线性回归模型通常将[**均方误差**](https://developers.google.cn/machine-learning/crash-course/glossary#MSE)用于损失函数，而逻辑回归模型则使用[**对数损失函数**](https://developers.google.cn/machine-learning/crash-course/glossary#Log_Loss)。

均方误差（MSE）：
$$
MSE = \frac{1}{N} \sum_{(x,y)\in D} (y - prediction(x))^2
$$

### 3.降低损失

 梯度下降算法中，下一点位置 = 梯度x学习速率（步长）

#### 迭代方法

![](img\google-1.jpg)

### 4.使用TF基础

合成特征：一种[**特征**](https://developers.google.cn/machine-learning/crash-course/glossary#feature)，不在输入特征之列，而是从一个或多个输入特征衍生而来



### 5.泛化

#### 判断模型出色与否

理论上：泛化理论（略）

直觉上：奥卡姆剃刀（越简单越好）

经验上：用测试集上的表现作为新数据的预测



前提假设：

- 样本独立同分布
- 分布是平稳的
- 始终从同一分布中抽取样本



### 6.验证

为了防止对测试集的过拟合，将数据集分为三个子集（新增一个验证集）

![](img\google-2.jpg)



### 7.表示法

#### 特征工程

定义：将原始数据转化为特征矢量

![](img\google-3.jpg)

对于一些无法直接转化为数字的数据（如字符串），通过映射，独热编码转化：

- 首先，为您要表示的所有特征的字符串值定义一个**词汇表**

- 然后，使用该词汇表创建一个**独热（one-hot）编码**（使用N位状态寄存器对N个状态进行编码），用于将指定字符串值表示为一个二元矢量。在该矢量（与指定的字符串值对应）中：

  - 只有一个元素设为 `1`。
  - 其他所有元素均设为 `0`。

  该矢量的长度等于词汇表中的元素数

对于分类值

分类特征具有一组离散的可能值，通常将每个分类特征表示为单独的`bool`值（是a吗？是b吗？），该方法同时有利于多分类情况

#### 良好特征具备的特性：

- 特征值以非零值形式在数据集中多次出现
- 特征应该具有清晰明确的含义
- 特征值的取值范围应该合理（如时间不要取-1）
- 特征的定义不随时间而变化（即平稳性）
- 分布不应包含离谱的离群值

#### 清理数据

**良好的机器学习依赖于良好的数据**，数据的重要性大于模型

##### 缩放特征值：

将浮点特征值从自然范围转换为标准范围

作用：

- 帮助梯度下降法更快收敛
- 帮助避免NaN陷阱（模型中的一个数字在训练期间变成 [NaN](https://en.wikipedia.org/wiki/NaN)，这会导致模型中的很多或所有其他数字最终也会变成 NaN）
- 帮助模型更好确定合适的权重

方法有：

- 线性映射
- 计算Z得分（scaled value = (value - mean) / stddev）

##### 处理极端离群值

- 取对数
- 设上限（可能造成上限出现峰值）

##### 分箱

某些特征值需要分箱处理后才能与标签值建立更好的预测模型（如房价与纬度没有线性关系，但在某一纬度范围内是可以预测的）

##### 清查

处理数据集时要将不可靠的样本给移出或修正



### 8.特征组合

定义：通过将两个或多个输入特征相乘来对特征空间中的**非线性**规律进行编码的合成特征

种类：如[A x B],[A x B x C],[A x A]

学习高度复杂模型：

- 对大规模数据集使用特征组合
- 神经网络




### 9.正则化(Regularization)：简单性

#### L2正则化

定义：通过降低复杂模型的复杂度来**防止过拟合**的原则称为正则化

为了避免训练集数据过拟合，应该求**结构风险最小化**（最小化损失和复杂度）：

$$
\text{minimize(Loss(Data|Model) + complexity(Model))}
$$
![](img\FireShot Capture 2 - 简化正则化 (Regularization for Simplicity)：_ - https___developers.google.cn_machin.png)

而衡量模型复杂度有两种常见方式：

- 用所有特征的权重的函数

  可以使用 **L2 正则化**公式来量化复杂度，该公式将正则化项定义为所有特征权重的平方和：

  L2 regularization term=||w||22=w12+w22+...+wn2
  $$
  L_2\text{ regularization term} = ||\boldsymbol w||_2^2 = {w_1^2 + w_2^2 + ... + w_n^2}
  $$

- 用具有非零权重的特征总数的函数




执行L2正则化对模型的影响：

- 使权重接近于0（并非正好为0）
- 使权重的平均值接近于0，且呈正态（钟形曲线或高斯曲线）分布

>L2 正则化降低较大权重的程度高于降低较小权重的程度。因此，即使某个权重降低的速度比另一个快，L2 正则化也往往会使较大权重降低的速度快于较小的权重。

#### Lambda

lambda又称**正则化率**

模型开发者通过以下方式来调整正则化项的整体影响：用正则化项的值乘以名为 **lambda**的标量

即：
$$
\text{minimize(Loss(Data|Model)} + \lambda \text{ complexity(Model))}
$$
增加lambda值将增强正则化的效果（直方图更剧烈），降低lambda值则会得到比较平缓的直方图

理想的 lambda 值取决于数据，因此您需要手动或自动进行一些调整



### 10.逻辑回归 (Logistic Regression)

#### 定义

一种模型，通过将 [**S 型函数**](https://developers.google.cn/machine-learning/crash-course/glossary#sigmoid_function)应用于线性预测，生成分类问题中每个可能的离散标签值的概率（有渐近线，可以提供(0,1)之间的有界值）。虽然逻辑回归经常用于[**二元分类**](https://developers.google.cn/machine-learning/crash-course/glossary#binary_classification)问题，但也可用于[**多类别**](https://developers.google.cn/machine-learning/crash-course/glossary#multi-class)分类问题（其叫法变为**多类别逻辑回归**或**多项回归**）。

#### S型函数的运用

如果 z 表示使用逻辑回归训练的模型的线性层的输出，则 S 型(z) 函数会生成一个介于 0 和 1 之间的值（概率）。用数学方法表示为：
$$
y' = \frac{1}{1 + e^{-(z)}}
$$

- y' 是逻辑回归模型针对特定样本的输出

- z 是 b + w1x1 + w2x2 + … wNxN，又称为对数几率，因为z的反函数为：
  $$
  z = log(\frac{y}{1-y})
  $$

  - w:该模型学习的权重和偏差
  - x:特定样本的特征值

![](img\FireShot Capture 3 - 逻辑回归 (Logistic Regression)：计算概率  I  机器_ - https___developers.google.cn_machin.png)

#### 损失函数

对数损失函数：
$$
Log Loss = \sum_{(x,y)\in D} -ylog(y') - (1 - y)log(1 - y')
$$

- (x,y)&straightepsilon;D 是包含很多有标签样本 (x,y) 的数据集。
- “y”是有标签样本中的标签。由于这是逻辑回归，因此“y”的每个值必须是 0 或 1。
- “y'”是对于特征集“x”的预测值（介于 0 和 1 之间）。

#### 逻辑回归中的正则化

因为逻辑回归的渐近性，如果不进行正则化，模型会因为促使所有样本的损失达到0但又做不到而造成过拟合，逻辑回归常常采用一下三种正则化方法：

- L1正则化
- L2正则化
- 早停法（即限制训练步数或步长）


### 11.分类

#### 真与假以及正类别与负类别

|                    | 正例（预测值） | 反例（预测值） |
| :----------------: | :------------: | :------------: |
| **正例（真实值）** |   真正例(TP)   |   假正例(FP)   |
| **反例（真实值）** |   假负例(FN)   |   真负例(TN)   |

正类别、负类别（Positive,Negative）：定义的样本中的二分类情况

真、假（True,False）:真——实际值与预测值相同，假——实际值与预测值相反

**真正例**是指模型将正类别样本正确地预测为正类别。同样，**真负例**是指模型将负类别样本正确地预测为负类别。

**假正例**是指模型将负类别样本错误地预测为正类别，而**假负例**是指模型将正类别样本错误地预测为负类别。

TP,TN不会造成损失

#### 准确率(Accuracy)

定义：预测正确的结果所占比例
$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

二分类问题中
$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
$$

#### 精确率和召回率（Precision & Recall）

#####  精确率

定义：表示的是预测为正的样本中有多少是真正的正样本，又称为**查准率**
$$
\text{Precision}= \frac{TP}{TP+FP}
$$

##### 召回率

定义：表示真正的正样本中有多少被正确预测，又称为**查全率**
$$
\text{Recall}= \frac{TP}{TP+FN}
$$

查准和查全二者之间是矛盾的。查准率高（尽可能只挑选有把握的），那查全率就会低；查全率高（试想把所有类别都预测为正例），那查准率就会低。

为了综合考虑二者，常常采用$F_1$值：
$$
F_1 = \frac{2}{\frac{1}{recall}+\frac{1}{precision}}=2 * \frac{precision*recall}{precision+recall}
$$

#### ROC和曲线下面积

真正例率	(TPR)：等同于召回率
$$
TPR = \frac{TP}{TP+FN}
$$
假正例率(FPR)：
$$
FPR = \frac{FP}{FP+TN}
$$

ROC 曲线用于绘制采用不同分类阈值时的 TPR 与 FPR



