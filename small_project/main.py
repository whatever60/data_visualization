import json
import pickle
import base64

import utils.pca as pca
import markdown

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors import KernelDensity

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go



with open('./mammoth_3d.json') as f:
    j = json.load(f)

with open('./embedding.pkl', 'rb') as f:
    embedding = pickle.load(f)

data = np.array(j)
data[:, 0] *= -1
data[:, [1,2]] = data[:, [2,1]]
distance = [sum((i - j) ** 2) for i, j in zip(data[1:], data[:-1])]
cut = np.sort(np.argpartition(distance, -7)[-7:] + 1)
color_num = [cut[0]] + (cut[1:] - cut[:-1]).tolist() + [len(data) - cut[-1]]
color_list = [
        '#55efc4',
        '#00cec9',
        '#0984e3',
        '#6c5ce7',
        '#b2bec3',
        '#fdcb6e',
        '#e17055',
        '#d63031',
        '#7f8c8d', 
        '#2c3e50'
    ]
color = []
for i, j in zip(color_list, color_num):
    color.extend([i] * j)


ALL_MODELS = [
    'Gaussian Mixture',
    'Gaussian Kernel',
    'Tophat Kernel',
    'Epanechnikov Kernel',
    'Exponential Kernel',
    'Linear Kernel',
    'Cosine Kernel'
]

KWARGS = [
    dict(model=mixture.GaussianMixture, n_components=8, covariance_type='full'),
    dict(model=KernelDensity, kernel='gaussian', bandwidth=0.2),
    dict(model=KernelDensity, kernel='tophat', bandwidth=0.2),
    dict(model=KernelDensity, kernel='epanechnikov', bandwidth=0.2),
    dict(model=KernelDensity, kernel='exponential', bandwidth=0.2),
    dict(model=KernelDensity, kernel='linear', bandwidth=0.2),
    dict(model=KernelDensity, kernel='cosine', bandwidth=0.2),
]

SLIDER_VALUE = [8] + [0.2] * 6

SLIDER_ARGS = [dict(min=1, max=15, step=1)] + [dict(min=0.1, max=2.1, step=0.1)] * 6

HINTS = [f'尝试改变{i}密度函数的{j}参数，当前参数值为{{}}' for i, j in zip(ALL_MODELS, ['n_components'] + ['bandwidth'] * 6)]


pca_coor = pca.mnist2array('./train-images.gz')
pca_label = pca.mnist2array('./train-labels.gz')
pca_label = np.array([str(i) for i in pca_label])
fig3d, fig2d, explained_variance = pca.pca(pca_coor, pca_label)

example_image = './example.png'
encoded_example = base64.b64encode(open(example_image, 'rb').read())


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Markdown(markdown.intro),
        dcc.Graph(
            figure=go.Figure(
                data=go.Scatter3d(
                        x=data[:, 0],
                        y=data[:, 1],
                        z=data[:, 2],
                        mode='markers',
                        marker=dict(color=color, size=2)
                )
            )
        ),
        dcc.Markdown(markdown.density1),
        dcc.Graph(
            figure=go.Figure(
                data=go.Scattergl(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(color=color, size=2)
                )
            )
        ),
        dcc.Markdown(markdown.density2),
        *[html.Button(n, id=n, n_clicks_timestamp=0) for n in ALL_MODELS],
        html.Div(id='slider-hint'),
        html.Div(
            id='the-slider',
            children=dcc.Slider(
                id='slider'
            )
        ),
        dcc.Graph(
            id='main_figure',
        ),
        dcc.Markdown(markdown.pca1),
            html.Img(
                src='data:image/jpg;base64,{}'.format(encoded_example.decode())
            ),
        dcc.Markdown(markdown.pca2),
        dcc.Graph(
            id='pca3d',
            figure=fig3d
        ),
        dcc.Markdown(markdown.pca3),
        dcc.Graph(
            id='pca2d',
            figure=fig2d
        ),
        dcc.Markdown(markdown.pca4),
        html.Img(
            id='number-image'
        ),
        dcc.Markdown(markdown.pca5),
        dcc.Graph(
            figure=go.Figure(
                data=go.Scatter(
                    x=np.arange(1, 785),
                    y=explained_variance / explained_variance.sum(),
                    mode='lines',
                )
            )
        ),
        dcc.Markdown(markdown.pca6)
    ],
    style={
        'width': '60%',
        'align-items': 'center',
        'margin': '0 auto',},
)


@app.callback(
    Output('number-image', 'src'),
    [
        Input('pca2d', 'clickData'),
        Input('pca3d', 'clickData')
    ]
)
def display_hover_data(data2d, data3d):
    dummy_data = np.ones((28,28)) * 255
    ctx = dash.callback_context
    if not ctx.triggered:
        data = dummy_data
    else:
        id_ = ctx.triggered[0]['prop_id'].split('.')[0]
        if id_ == 'pca2d':
            data = data2d
        elif id_ == 'pca3d':
            data = data3d
        else:
            data = dummy_data
        try:
            data = pca_coor[data['points'][0]['customdata'][0]].reshape(28, 28)
        except:
            data = dummy_data
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    plt.savefig('./cached.png')
    cached_image = './cached.png' # replace with your own image
    cached_example = base64.b64encode(open(cached_image, 'rb').read())
    return 'data:image/png;base64,{}'.format(cached_example.decode())


@app.callback(
    [
        Output(component_id='main_figure', component_property='figure'),
        # Output(component_id='main_figure', component_property='className'),
        Output(component_id='the-slider', component_property='children'),
        Output(component_id='slider-hint', component_property='children')
    ],
    [
        Input(component_id='slider', component_property='value'),
        *[Input(component_id=n, component_property='n_clicks_timestamp') for n in ALL_MODELS],
    ],
)
def update_main_figure(value, *timestamps):
    ctx = dash.callback_context
    if not ctx.triggered:
        id_ = ALL_MODELS[0]
    else:
        id_ = ctx.triggered[0]['prop_id'].split('.')[0]
    changed_id = np.argmax(np.array([int(t) for t in timestamps]))
    if id_ == 'slider':
        SLIDER_VALUE[changed_id] = value
        kwargs = KWARGS[changed_id]
        if 'bandwidth' in kwargs:
            kwargs['bandwidth'] = value
        else:
            kwargs['n_components'] = value
        return (
            draw(**kwargs),
            dcc.Slider(
                id='slider', value=SLIDER_VALUE[changed_id], **SLIDER_ARGS[changed_id]
            ),
            HINTS[changed_id].format(SLIDER_VALUE[changed_id])
        )
    else:
        return (
            draw(**KWARGS[changed_id]),
            dcc.Slider(
                id='slider', value=SLIDER_VALUE[changed_id], **SLIDER_ARGS[changed_id]
            ),
            HINTS[changed_id].format(SLIDER_VALUE[changed_id])
        )


def draw(model, **kwargs):
    model2d = model(**kwargs).fit(embedding)
    modelx = model(**kwargs).fit(embedding[:, 0].reshape(-1, 1))
    modely = model(**kwargs).fit(embedding[:, 1].reshape(-1, 1))
    
    x, y = np.linspace(-20, 30, 101), np.linspace(-20, 25, 91)
    
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -model2d.score_samples(XX)
    Z = np.log2(Z.reshape(X.shape))

    den0 = modelx.score_samples(x.reshape(-1, 1))
    den1 = modely.score_samples(x.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Contour(
            z=np.nan_to_num(Z, posinf=6),
            dx=0.5,
            dy=0.5,
            x0=-20,
            y0=-20,
            contours_coloring='heatmap',
            colorscale='ice',
            colorbar=dict(
                tickvals=np.arange(2, 10),
                ticktext=[i.format(2 ** val) for val, i in enumerate(['10^-{}'] * 8, 2)]
            )
        )
    )
    fig.add_trace(go.Scattergl(
            x = embedding[:, 0],
            y = embedding[:, 1],
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = color,
                size = 2
            )
        ))
    fig.add_trace(go.Scatter(
            y=y,
            x=np.exp(den1),
            xaxis = 'x2',
            mode='lines',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))
    fig.add_trace(go.Scatter(
            x=x,
            y=np.exp(den0),
            yaxis = 'y2',
            mode='lines',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))

    fig.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False,
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False,
            range=[-20, 25]
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        height = 800,
        width = 800,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)