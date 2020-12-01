intro = '''
    # 2020秋可视化平时作业

    曲一鸣 2018012374

    选题是density map和PCA

    使用dash实现。dash的后端是flask，前端是react和plotly，plotly底层是d3

    # Density Map

    ### 太长不看

    将三维的猛犸象散点图，调用降维算法UMAP降维至二维，然后绘制density map

    ### 数据来源
    https://github.com/PAIR-code/understanding-umap/blob/master/raw_data/mammoth_3d.json
'''

density1 = '''
    ### 降维算法UMAP

    2018年由Mclnnes在UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction中提出

    特点是快，而且能在低维空间中更好地保留高维空间的全局结构特征
    
    其他领域不知道，但在我了解的测序数据分析领域已经迅速取代了t-SNE

    背后的数学比t-SNE复杂不少，[这里](https://pair-code.github.io/understanding-umap/)有个不错的简介

    将上面三维的猛犸象用umap降维到二维平面，使用默认参数

    ```python
    import umap
    import plotly.graph_objects as go


    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x, y ,z)
    go.Figure(
        data=go.Scattergl(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(color=color, size=2)
        )
    ```
'''

density2 = '''
    在二维平面上依然可以大概辨认猛犸象的各个身体结构

    ### 密度估计

    使用七种估计方法。第一种混合高斯模型，虽然经常用于聚类，但是实质上还是密度估计。有一个参数n_components，用于指定混合模型中高斯分布的个数
    
    ./utils/gmm.py中有自己用numpy实现的高斯混合模型，结果和sklearn是一样的，但没有优化所以速度比sklearn慢一点，这里用的是sklearn版本

    另外六种是核函数，即sklearn中提供的Gaussian，tophat，Epanechnikov，linear和cosine，它们都有bandwidth参数，可以衡量以每个点为中心的分布的方差，或每个点影响力的分散程度

    这些核函数没有手动实现，直接调用sklearn，一方面因为直接实现起来比较简单，只是许多分布函数求和，另一方面sklearn的核函数做了优化（https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html），速度更快
    
    当前展示的是混合高斯模型的密度估计，可以通过下面的按钮和滑动条改变密度函数并设置参数，分布的计算和绘制需要几秒钟时间

    计算出的密度先以10为底取log，取相反数，再以2为底取log后呈现在图上，所以要注意热力图颜色的含义，可以参考右侧标尺的刻度

    热力图绘制细节：横纵坐标每隔0.5取点，形成一个网格，这些点在分布上采样，然后平滑处理（可以通过绘图函数的参数设置）得到热力图

    可以看出带宽越大，每个点影响范围越广，但影响力也越分散，整体分布越趋于均匀分布

    核函数的选取取决于具体问题，有的核函数只能趋于0，有些核函数会取到0，这些核函数在绘制热力图时将所有0值设置为10^-64，主要出于美观，不至于大片区域是空白
'''

pca1 = '''
    # PCA

    PCA的想法很有启发性但步骤不多，./utils/pca.py中有numpy版本，下面使用的也是该版本。

    实现过程参考了https://dev.to/akaame/implementing-simple-pca-using-numpy-3k0a

    ### 太长不看

    将784维的MNIST数据集使用PCA降维到三维或二维

    ### 数据来源

    MNIST训练集的60000个图片

    http://yann.lecun.com/exdb/mnist/

    ```python
    # 读入数据 https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    import struct
    import gzip
    
    import numpy as np
    import matplotlib.pyplot as plt

    def mnist2array(filename):
        with gzip.open(filename) as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    data = mnist2array('./train-images.gz')
    label = mnist2array('./train-labels.gz')

    plt.imshow(data[0], cmap='gray')
    ```
'''

pca2 = '''
    ### 降维算法PCA
    
    这里就简单一点，不保留图片空间信息，直接把图片当做一维向量

    ```python
    data.shape = (60000, -1)
    ```

    每个图片是行向量，每列代表某个维度的像素值，所以需要列归一化

    ```python
    data = data - data.mean(axis=0)
    std = data.std(axis=0)
    data /= np.where(std, std, 1)
    ```

    计算协方差矩阵

    ```python
    cov = data.T @ data / data.shape[0]
    ```

    协方差矩阵的特征值分解，并按照特征值排序
    
    ```python
    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1]
    v = v[idx]
    w = w[:, idx]
    ```

    特征向量就是计算出的投影方差之和最大的方向。按照特征值排序后取前几个，就得到低维空间的坐标轴，该低维空间就保留高维空间的方差而言是最优的
    
    把高维向量投影到低维空间

    ```python
    axis = w[:, :3]
    x, y, z = (data @ axis).T
    ```

    三维投影

    ```python
    import plotly.graph_objects as go


    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(color=label, size=2)
    )])
    fig.show()
    ```
'''

pca3 = '''
    同理可以降到二维
'''

pca4 = '''
    点击数据点可以在下方查看对应的图片
'''

pca5 = '''
    可以发现降维后的数据一团糟，说明PCA的出发点不像t-SNE或umap，PCA只是减少维度，不是用来可视化的，如果强行降到人可读的维度，虽然最大程度保留了方差信息，但人无法从中获取见解。
    
    最后展示一下每个主成分多大程度上代表了原始数据的方差
'''

pca6 = '''
    可以看到，对于MNIST来说，确实有大量维度的冗余，但是第一个主成分也只占5%的方差，合适的维度应该至少不低于100，强行降到人可视的维度确实欠妥。
'''
