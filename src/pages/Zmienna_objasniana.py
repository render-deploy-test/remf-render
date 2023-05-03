import dash
from dash import html, dcc, callback, Input, Output,dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
from assets.data_loader import DataLoader
from assets.graphs import CreateGraph

dash.register_page(__name__, order=2)
abs_corr = pd.read_csv('../src/assets/abs_corr.csv') # do poprawy później
df_train = pd.read_csv('../src/assets/train.csv')
df = df_train.copy()
cg = CreateGraph(df_train)
korelacja = cg.Y_corr(abs_corr)
rozklad_cena = cg.dash_hist_with_normal_curve()
rozklad_log = cg.dash_hist_with_normal_curve(log=True)
scatter = cg.plot_scatter()
Y_train = df_train['SalePrice'].copy()
to_del = ['SalePrice', "Id"]
df_train = df_train.drop(to_del, axis=1)
drop_style = {'background-color': '#cfa527', 'textAlign': 'center', 'margin': 'auto', 'color':'black'}
drop_scatter = dcc.Dropdown(id='drop-scatter', options=[{"label":i, "value":i} for i in df_train.columns],
                        placeholder='Wybierz zmienna do analizy', className='dropdown',
                        style=drop_style)
layout = html.Div([
# tytul

    html.Div([
    dcc.Tabs(
        id='tabs-2',
        children=[
            dcc.Tab(label='Zmienna Objaśniana', value='tab-1',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='Po zastosowaniu transformacji logarytmicznej', value='tab-2',style = {'color':'white'},selected_style ={"background":'#035891'}),
        ],
        value='tab-1',
    colors={
        "border":"#242424", #obwodka
        "background":'#242424', #tlo
        'primary':'#035891' #jesli wybrane

    },
),
    html.Div(id='div-2')
])])

@callback(
    Output('div-2', 'children'),
    [Input('tabs-2', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
# statystyki opisowe
html.Div([
        html.Div([
            html.H6(children='Skośność',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
                html.H6(children='1.88',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Kurtoza',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='6.54',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Mediana ',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='$ 163 000',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Średnia ceny sprzedaży',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children="$ 180 921",
                    style={'textAlign': 'center',
                           'color': 'white'}),

        ], className='card_container three columns'),],className='row flex display'),
html.Div([
    html.Div([
        dcc.Graph(figure=korelacja),

    ],className='add_container six columns'),
    html.Div([
        dcc.Graph(figure=rozklad_cena),

    ], className='add_container six columns')

]),


])
    elif tab == 'tab-2':
        return html.Div([
# statystyki opisowe
html.Div([
        html.Div([
            html.H6(children='Skośność',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
                html.H6(children='0.12',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Kurtoza',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='0.81',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Mediana ',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='12',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Średnia ceny sprzedaży',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children="12.02",
                    style={'textAlign': 'center',
                           'color': 'white'}),

        ], className='card_container three columns'),],className='row flex display'),
html.Div([

    html.Div([
        html.Br(),
        dcc.Graph(figure=rozklad_log),
        html.Br()

    ], className='add_container six columns'),

    html.Div([
        html.H6(children='Analiza atrybutów - przed transformacja log',
                style={'textAlign': 'center',
                       'color': '#616161'}
                ),
        drop_scatter,
        html.Br(),
        dcc.Graph(id="graph-scatter-1", figure=scatter)
    ],
        className='add_container six columns')

]),


])
@callback(
    Output("graph-scatter-1", "figure"),
    [Input("drop-scatter", "value")]
)
def update_graph(x):
    fig = px.scatter(df, x=x, y='SalePrice')
    fig.update_layout(yaxis=dict(showgrid=False, title='Cena Sprzedazy',zeroline=False),
                      xaxis=dict(showgrid=False, title=x, zeroline=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#616161', title_font={"size": 20},width=600, height=390)
    return fig




