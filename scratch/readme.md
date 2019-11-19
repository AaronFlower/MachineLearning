### Regression

- [ ] Linear Regression
- [ ] High-order Polynomial
- [ ] FM
- [ ] DeepFM
- [ ] CART
- [ ] Boost
- [ ] SVM

### Classification
- [ ] LR
- [ ] High-Order Polynomial
- [ ] FM
- [ ] DeepFM


### Optimizer

- [ ] SGD
- [ ] AdaGrad
- [ ] Adam
- [ ] RMSProp
- [ ] Ftrl


### About sklearn

sklearn Provides a dataset tool to generate datasets for both classification and regression.

```python
import sklearn

sklearn.datasets.make_classification()
sklearn.datasets.make_regression()
```

There are many other methods to generate datasets, though the two methods above are most common.

### Why?

#### 1. 为什么要进行独热编码？

大部分机器学习算法会为每一个特征学习一个权重参数，或者需要特征来度量样本之间的距离。前者的情况如，Logistic Regression.

假设我们的数据集仅有一个 "Nationality" 特征：

```python
[["UK"], ["USA"], ["French"]]
```

如果不进行独热编码的话，我们一般会将它们编码成:

```python
[[0], [1], [2]]
```

那学习到模型就是: `y = w*X + b`, y > 0 或 y < 0; 这个模型不可能让你做一个三路选择。

而进行 One-hot 后，我们可以学习到 3 个参数，就可以实现三路选择了。

再者，在基于距离的机器学习算法中（如：kNN, k-means)，不进行 one-hot 的样本在计算距离时会有偏差。
如上面的例子：d(UK, USA) = 1, d(UK, French) = 2
而进行 one-hot 后：

```python
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```
d(UK, USA) = d(UK, French) = sqrt(2)

顺便说下：在决策树或其延伸算法（随机森林）中，如果树的尝试是足够的。那么是可以不用 one-hot 的，决策树可以学习到合适的参数 。

1. [Why does one hot encoding improve machine learning performance?](https://stackoverflow.com/questions/17469835/why-does-one-hot-encoding-improve-machine-learning-performance)




