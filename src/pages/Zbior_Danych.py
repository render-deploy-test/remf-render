import dash
from dash import html, dcc,callback, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from assets.data_loader import DataLoader
from assets.graphs import CreateGraph

dash.register_page(__name__, order=1)

data_loader = DataLoader(train_file="../src/assets/train.csv", test_file="../src/assets/test.csv")
df_train, df_test = data_loader.load_data()
df_all = pd.concat([df_train, df_test], ignore_index=True, sort=False)
# inicjacja obiektów klasy CreateGraph
cg = CreateGraph(df_train)
cg_test = CreateGraph(df_test)
cg_all = CreateGraph(df_all)

# dla treningowego
fig_ = cg.dash_miss_data(19)
fig = cg.dash_corr()
# dla testowego
fig_test = cg_test.dash_miss_data(33)
fig__test = cg_test.dash_corr()
#dla calego
fig_all = cg_all.dash_corr()
fig__all = cg_all.dash_miss_data(33)
layout = html.Div([
# tytul

    html.Div([
    dcc.Tabs(
        id='tabs-3',
        children=[
            dcc.Tab(label='Zbiór treningowy', value='tab-1',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='Zbiór testowy', value='tab-2',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='Cały zbiór', value='tab-3',style = {'color':'white'},selected_style ={"background":'#035891'}),
        ],
        value='tab-1',
    colors={
        "border":"#242424", #obwodka
        "background":'#242424', #tlo
        'primary':'#035891' #jesli wybrane

    },
),
    html.Div(id='div-3')
])])
@callback(
    Output('div-3', 'children'),
    [Input('tabs-3', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
html.Div([
        html.Div([
            html.H6(children='Nazwa zbioru',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
                html.H6(children='Ames',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Liczba instancji',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='1460',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Liczba atrybutów',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children='80',
                    style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container three columns'),

        html.Div([
            html.H6(children='Średnia ceny sprzedaży',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.H6(children="$ "+str(int(df_train.SalePrice.mean())),
                    style={'textAlign': 'center',
                           'color': 'white'}),

        ], className='card_container three columns'),],className='row flex display'),
html.Div([
    html.Div([
        dcc.Graph(figure=fig_),

    ],className='add_container five columns'),
    html.Div([
        dcc.Graph(figure=fig),

    ], className='add_container seven columns')

]),



])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([
                    html.H6(children='Nazwa zbioru',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='Ames',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='Liczba instancji',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='1460',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='Liczba atrybutów',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='80',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns')
, ], className='row flex display'),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_test),

                ], className='add_container five columns'),
                html.Div([
                    dcc.Graph(figure=fig__test),

                ], className='add_container seven columns')

            ]),

        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div([
                html.Div([
                    html.H6(children='Nazwa zbioru',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='Ames',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='Liczba instancji',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='2920',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='Liczba atrybutów',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='80',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                 ], className='row flex display'),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig__all),

                ], className='add_container five columns'),
                html.Div([
                    dcc.Graph(figure=fig_all),

                ], className='add_container seven columns')

            ]),

        ])
