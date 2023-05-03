import pandas as pd
import scipy
import plotly.graph_objects as go

class DataAnalysis:
    def __init__(self,df):
        self.df = df
        self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    def check_skewness(self):
        num_ = self.df.select_dtypes(include=self.numerics)
        skew_num = num_.apply(lambda x: scipy.stats.skew(x))
        skew_num = skew_num[skew_num > 1].sort_values(ascending=False)
        return pd.DataFrame(skew_num)
    def plot_skewness(self, title = 'Skośność atrybutów numerycznych większa od 1', x_title = 'Wartość'):
        x = self.check_skewness()
        data = go.Bar(x=x[0].round(2), y=x[[0]].index, orientation='h',
                      text=x[0].round(2), marker_color='red')
        data = [data]
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title, title_x=0.5,
            hovermode="y unified",
            xaxis_title=x_title, yaxis_title="Nazwa atrybutu numerycznego",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=25, color="black", family="Lato, sans-serif"),
            font=dict(color="black", size=16),
            showlegend=False, width=1500, height=900,
            hoverlabel=dict(bgcolor="black", font_size=13, font_family="Lato, sans-serif"))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, showline=False, automargin=True, autorange=True,
                         categoryarray=[x.index])
        fig.update_traces(marker=dict(line=dict(width=1)),textposition='inside', textangle =0)
        fig.layout.bargap = 0.5
        return fig.show()