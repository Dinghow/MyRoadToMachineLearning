### 12. 正则化：稀疏性

稀疏矢量的特征组合通常会导致包含更多无意义维度，在降低模型复杂度时，我们希望将一些无意义的权重设为0（即L0正则化），但是L0正则化是非凸优化，并存在其他问题，所以我们将条件放宽至L1正则化，减小权重的绝对值，即采用L1正则化

> 凸优化：不严格的说，凸优化就是在标准优化问题的范畴内，要求目标函数和约束函数是凸函数的一类优化问题。

#### L1正则化

##### L1 和 L2 正则化

L2 和 L1 采用不同的方式降低权重：

- L2 会降低权重2。
- L1 会降低 |权重|。

因此，L2 和 L1 具有不同的导数：

- L2 的导数为 2 * 权重。
- L1 的导数为 k（一个常数，其值与权重无关）

即

- L2正则化永远不回将权重变为0
- L1正则化可以使权重为0: L1 在 0 处具有不连续性，这会导致与 0 相交的减法结果变为 0。例如，如果减法使权重从 +0.1 变为 -0.2，L1 便会将权重设为 0



### 13. 神经网络

#### 基本概念

之前处理特殊的非线性问题时，我们采用了特征组合的方法，但是对于一般的非线性问题，我们采用神经网络来进行模型的建立。

> 神经网络(neural network)：一种模型，灵感来源于脑部结构，由多个层构成（至少有一个是[**隐藏层**](https://developers.google.cn/machine-learning/crash-course/glossary#hidden_layer)），每个层都包含简单相连的单元或[**神经元**](https://developers.google.cn/machine-learning/crash-course/glossary#neuron)（具有非线性关系）。
>
> 隐藏层(hidden layer)：[**神经网络**](https://developers.google.cn/machine-learning/crash-course/glossary#neural_network)中的合成层，介于[**输入层**](https://developers.google.cn/machine-learning/crash-course/glossary#input_layer)（即特征）和[**输出层**](https://developers.google.cn/machine-learning/crash-course/glossary#output_layer)（即预测）之间。神经网络包含一个或多个隐藏层。

![](https://github.com/Dinghow/MyRoadToMachineLearning/raw/master/note/img/google-7.png)

#### 激活函数

这样添加隐藏层后，线性函数与线性函数的组合依然是线性的，所以我们通过添加激活函数（非线性转化层）来使模型可以处理非线性问题。

![](https://github.com/Dinghow/MyRoadToMachineLearning/raw/master/note/img/google-8.png)

> 激活函数 (activation function)：一种函数（例如 [**ReLU**](https://developers.google.cn/machine-learning/crash-course/glossary#ReLU) 或 [**S 型**](https://developers.google.cn/machine-learning/crash-course/glossary#sigmoid_function)函数），用于对上一层的所有输入求加权和，然后生成一个输出值（通常为非线性值），并将其传递给下一层。
>
> ![](https://github.com/Dinghow/MyRoadToMachineLearning/raw/master/note/img/google-9.png)
>
> ReLU函数：修正线性单元(Rectified Linear Unit)，规则为：F(x)=max(0,x)

激活函数的引入，在非线性上堆叠非线性，使得模型可以处理更为复杂的输出预测

实际上所有的函数都可以作为激活函数，我们假设激活函数 $\sigma$ ，网络中节点的值为 $v$ :
$$
v = \sigma(\boldsymbol w \cdot \boldsymbol x+b)
$$

#### 组件

神经网络包括：

- 一组节点$v$，类似于神经元，位于层中。
- 一组权重$w$，表示每个神经网络层与其下方的层之间的关系。下方的层可能是另一个神经网络层，也可能是其他类型的层。
- 一组偏差$b$，每个节点一个偏差。
- 一个激活函数$\sigma$ ，对层中每个节点的输出进行转换。不同的层可能拥有不同的激活函数。



## 14. 训练神经网络

最常见的神经网络训练算法：[反向传播算法](https://google-developers.gonglchuangl.net/machine-learning/crash-course/backprop-scroll/)（BP算法，BackPropagation）

> 是一种与[最优化方法](https://zh.wikipedia.org/wiki/%E6%9C%80%E4%BC%98%E5%8C%96)（如[梯度下降法](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)）结合使用的，用来训练[人工神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)的常见方法。该方法对网络中所有权重计算[损失函数](https://zh.wikipedia.org/w/index.php?title=%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0&action=edit&redlink=1)的梯度。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。

很多情况都会导致BP算法出错

- 梯度消失：较低层（更接近输入）的梯度可能会变得非常小。在深度网络中，计算这些梯度时，可能涉及许多小项的乘积。当较低层的梯度逐渐消失到 0 时，这些层的训练速度会非常缓慢，甚至不再训练。

  ReLU 激活函数有助于防止梯度消失。

- 梯度爆炸：如果网络中的权重过大，则较低层的梯度会涉及许多大项的乘积。在这种情况下，梯度就会爆炸：梯度过大导致难以收敛。

  批标准化可以降低学习速率，因而有助于防止梯度爆炸。

- ReLu单元消失：一旦 ReLU 单元的加权和低于 0，ReLU 单元就可能会停滞，梯度无法反向传播。由于梯度的来源被切断，ReLU 的输入可能无法作出足够的改变来使加权和恢复到 0 以上。

  降低学习速率有助于防止 ReLU 单元消失。



丢弃正则化

> 一种形式的[**正则化**](https://developers.google.cn/machine-learning/crash-course/glossary#regularization)，在训练[**神经网络**](https://developers.google.cn/machine-learning/crash-course/glossary#neural_network)方面非常有用。丢弃正则化的运作机制是，在神经网络层的一个梯度步长中移除随机选择的固定数量的单元。丢弃的单元越多，正则化效果就越强。



## 15. 多类别神经网络

### 一对多

一对多提供了一种利用二元分类的方法，对于N个可行的解决方案，就包括N个单独的二元分类器

![](https://github.com/Dinghow/MyRoadToMachineLearning/raw/master/note/img/google-10.png)

### Softmax

> Softmax函数，又称为归一化函数，它能将一个含任意实数的K维向量 $Z$ “压缩”到另一个K维实向量 $\sigma(z)$中，使得每一个元素的范围都在之间$(0,1)$，并且所有元素的和为1。

根据Softmax函数的这一特点，常常将其用多类别领域，Softmax为每个类别分配一个小数表示概率，其和为1

Softmax函数的表达式为：
$$
p(y = j|\textbf{x})  = \frac{e^{(\textbf{w}_j^{T}\textbf{x} + b_j)}}{\sum_{k\in K} {e^{(\textbf{w}_k^{T}\textbf{x} + b_k)}} }
$$
而在神经网络中，Softmax层必须有和输出层一样的节点数

#### Softmax的选项

- 完整Softmax：针对所有类别都要计算概率
- 候选采样：针对正类别标签计算概率，针对负类别标签随机采样计算概率

> **候选采样：**在训练一个多类别分类器过程中，所有的类别都需要进行评估。为了解决这个问题，人们发明了候选采样的技巧，每次只评估所有类别的一个很小的子集。tensorflow包含了多种候选采样函数

#### Softmax不适用的情况

对于每一个样本可能是多个类别的成员是，Softmax不适用，必须依赖于多个逻辑回归



## 16. 嵌入

