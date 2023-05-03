import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class CreateGraph:
    def __init__(self, df):
        self.df = df
    def hist_with_normal_curve(self):
        """
        Metoda zwraca rozkład zmiennej z
        :return: figure
        """

        fig = ff.create_distplot(
            [self.df["SalePrice"].tolist()],
            group_labels=["SalePrice"],
            show_hist=False,
            colors=['red'],
            curve_type="normal"
        ).add_traces(
            px.histogram(self.df['SalePrice'], x="SalePrice", nbins=80, color="SalePrice",
                         color_discrete_sequence=['#1f77b4'])
            .update_traces(yaxis="y3", name="histogram")
            .data
        ).update_layout(yaxis3={"overlaying": "y", "side": "left"},
                        showlegend=False)  # template='plotly_dark' - ciemne tło
        fig.update_layout(title_text='Rozkład zmiennej SalePrice', title_x=0.5, title_font_size=30, xaxis_title="Cena",
                          yaxis_title="Częstotliwość", font=dict(color='black'), paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', yaxis_visible = False)
        fig["layout"].pop("updatemenus")
        fig.update_yaxes(showgrid=False)
        return fig.show()

    def dash_corr(self, to_drop=['YrSold', 'MoSold', 'MSSubClass', 'GarageYrBlt']):
        """
        Metoda zwraca wykres macierzy korelacji
        :param to_drop: zmienne, które maja byc niewidoczne na wykresie
        :return: figure
        """
        df = self.df.drop(to_drop, axis=1).corr().abs()
        fig = px.imshow(df, title="Macierz korelacji", color_continuous_scale="delta")
        fig.layout.height = 600
        fig.layout.width = 600
        fig.update_layout(title_font={'size': 35, 'color': '#616161'}, title_x=0.5, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font_color="#94cce3")
        return fig
    def dash_miss_data(self, x):
        """
        Metoda zwraca histogram dot. braków danych w zbiorze danych.
        :param x: liczba zmiennych do wyświetlenia na wykresie (w kolejności malejacej)
        :return: figure
        """
        miss_data = go.Bar(
            y=self.df.isnull().sum().sort_values(ascending=False).head(x).index,
            x=self.df.isnull().sum().sort_values(ascending=False).head(x),
            text=self.df.isnull().sum().sort_values(ascending=False).head(x),
            orientation='h'
        )
        missing_data = [miss_data]
        layout = go.Layout(
            title='Brakujące dane',
            yaxis=dict(title='Nazwa atrybutu', showgrid=False, title_font={"size": 20}),
            xaxis=dict(title='Liczba brakujących danych', showgrid=False, title_font={"size": 20, 'color': 'white'}),
            font_color='white'
        )
        fig_ = go.Figure(data=missing_data, layout=layout)
        fig_.layout.height = 600
        fig_.layout.width = 480
        fig_.update_traces(marker_color='red',
                           marker_line_color='black',
                           marker_line_width=1, opacity=0.6, texttemplate='%{text:.4s}',
                           textposition='auto', constraintext='both')
        fig_.update_layout(title_font={'size': 35, 'color': '#616161'}, title_x=0.5,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        fig_.update_yaxes(showgrid=False, showline=False, automargin=True, autorange=True,
                          categoryarray=self.df.isnull().sum().sort_values(ascending=False).head(19).index,
                          categoryorder='array', color='white')
        fig_.update_xaxes(color='white')
        return fig_

    def Y_corr(self, data):
        """
        Metoda zwraca histogram przedstawiajacy korelacje zmiennych objasniajacych z zmienna objasniana
        :param data: dataframe zawierajacy 2 kolumny - zmienna zalezna oraz wartość jej korelacji z Y
        :return: figure
        """
        abs_corr = pd.DataFrame(data)
        abs_corr.SalePrice = abs_corr.SalePrice.sort_values(ascending=False)
        abs_corr_text = abs_corr
        data = go.Bar(x=abs_corr['SalePrice'], y=abs_corr['index'], orientation='h',
                      text=abs_corr_text, marker_color='red')
        data = [data]
        fig = go.Figure(data=data)
        fig.update_layout(
            title='Korelacja atrybuty numeryczne', title_x=0.4,
            hovermode="y unified",
            xaxis_title='Wartość bezwzględna współczynnika korelacji ',  # yaxis_title="Nazwa atrybutu numerycznego",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=20, color='#035891', family="Lato, sans-serif"),
            font=dict(color="#616161", size=12),
            showlegend=False, width=600, height=450,
            hoverlabel=dict(bgcolor="black", font_size=13, font_family="Lato, sans-serif"))
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showline=False, automargin=True, autorange=True,
                         categoryarray=[abs_corr.index], zeroline=False)
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.layout.bargap = 0.5
        return fig
# to samo co pierwsza metoda do weryfikacji
    def dash_hist_with_normal_curve(self, log = False, Y_name='SalePrice'):
        """
        Metoda zwracajaca rozkład zmiennej.
        :param df: dane wejściowe
        :param log: czy zmienna objaśniana ma zostać poddana transformacji logarytmicznej - True lub False.
        Domyślnie False.
        :param Y_name:Nazwa zmiennej, której rozkład chcemy otrzymać - podana w formacie string
        :return:figure
        """

        df = self.df.copy()
        if log == True:
            df[Y_name] = np.log1p(df[Y_name])
        fig = ff.create_distplot(
            [df[Y_name].tolist()],
            group_labels=[Y_name],
            show_hist=False,
            colors=['red'],
            curve_type="normal"
        ).add_traces(
            px.histogram(df[Y_name], x=Y_name, nbins=80, color=Y_name,
                         color_discrete_sequence=['#1f77b4'])
            .update_traces(yaxis="y3", name="histogram")
            .data
        ).update_layout(yaxis3={"overlaying": "y", "side": "left"},
                        showlegend=False)  # template='plotly_dark' - ciemne tło
        fig.update_layout(title_text='Rozkład zmiennej {0}'.format(Y_name), title_font_color='#035891', title_x=0.5,
                          title_font_size=20, xaxis_title="Cena",
                          yaxis_title="Częstotliwość", font=dict(color='#616161'), paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', width=600, height=450,yaxis_visible = False)
        fig["layout"].pop("updatemenus")
        fig.update_yaxes(showgrid=False, zeroline=False)
        return fig
    def compare_with_log(self, Y_name = 'SalePrice'):
        """
        Metoda zwraca Dataframe zawierajacy 2 kolumny - przedstawiajace zmienne przed i po transformacji logarytmicznej
        :param Y_name: Nazwa zmiennej, której rozkład chcemy otrzymać - podana w formacie string
        :return: DataFrame
        """
        log = "{0}_log".format(Y_name)
        df = self.df.copy()
        df[log] = np.log1p(df[Y_name]).round(4)
        comparison = df[[Y_name, log]]
        return pd.DataFrame(comparison)

    def plot_scatter(self, x='MSSubClass'):
        fig = px.scatter(self.df, x=x, y='SalePrice')
        fig.update_layout(yaxis=dict(showgrid=False, title='Cena Sprzedazy', zeroline=False),
                          xaxis=dict(showgrid=False, title=x, zeroline=False),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#616161', title_font={"size": 20}, width=600, height=390)
        return fig



