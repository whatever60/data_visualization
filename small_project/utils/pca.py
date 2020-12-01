import struct
import gzip

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn import decomposition


def mnist2array(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def pca(data, label):
    dim = 3

    data.shape = (60000, -1)

    data = data - data.mean(axis=0)
    std = data.std(axis=0)
    data /= np.where(std, std, 1)
    cov = data.T @ data / data.shape[0]

    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1]
    v = v[idx]
    w = w[:, idx]

    axis = w[:, :dim]
    coor = data @ axis

    df = pd.DataFrame(
        data=np.hstack((coor, label[:, np.newaxis])),
        columns=('1st_principal', '2nd_principal', '3rd_principal', 'label')
    )
    df['size']=np.ones(60000) * 0.2
    df['idx'] = np.arange(60000)
    fig3d = px.scatter_3d(df, x='1st_principal', y='2nd_principal', z='3rd_principal', color='label', size='size', custom_data=['idx'])
    fig2d = px.scatter(df, x='1st_principal', y='2nd_principal', color='label', size='size', custom_data=['idx'])
    fig3d.update_traces(marker=dict(size=2))
    fig2d.update_traces(marker=dict(size=3))

    return fig3d, fig2d, v


def sk_pca(dim, data, label):
    data.shape = (60000, -1)

    pca = decomposition.PCA()
    pca.n_components = dim
    pca_data = pca.fit_transform(data)
    coor = pca_data[:, :dim]
    
    if dim == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=coor[:, 0],
            y=coor[:, 1],
            z=coor[:, 2],
            mode='markers',
            marker=dict(color=label, size=2)
        )])
    else:
        fig = go.Figure(data=[go.Scattergl(
            x=coor[:, 0],
            y=coor[:, 1],
            mode='markers',
            marker=dict(color=label, size=2)
        )])
    return fig
