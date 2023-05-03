import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import pandas as pd
import dalex as dx
from assets.DashExplainers import DashExplainers
import plotly


# przypisanie nazw modeli
xgb = "XGBoost"
ada = "AdaBoost"
rf = "Lasy Losowe"
dt = "Drzewo Decyzyjne"
# wczytanie danych
ce = DashExplainers('../src/assets/train.csv', '../src/assets/test.csv')
df, df_test = ce.load_data()
num_cols = df.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
#load explainers
exp_xgb = ce.load_explainers("../src/assets/explainer_XGBoost.pkl",xgb)
exp_ada = ce.load_explainers("../src/assets/explainer_AdaBoost.pkl", ada)
exp_rf = ce.load_explainers("../src/assets/explainer_Lasy Losowe.pkl", rf)
exp_dt = ce.load_explainers("../src/assets/explainer_Drzewo Decyzyjne.pkl", dt)
# generowanie wykresów istotności zmiennych
# ce.plot_ALE()
# ce.plot_importance()
figures = ce.load_plots(names = [xgb,ada,rf,dt])
# definicja dropdownów
drop_style = {'background-color': '#cfa527', 'textAlign': 'center', 'margin': 'auto', 'color':'black'}
drop_ALE = dcc.Dropdown(id='drop-1', options=[{"label":i, "value":i} for i in num_cols],
                        placeholder='Wybierz zmienna do analizy', className='dropdown',
                        style=drop_style)
drop_importance = dcc.Dropdown(id='drop-2', options=[{"label":i,"value":i } for i in num_cols], multi=True,
                                 placeholder='Wybierz zmienne do analizy',className='dropdown',
                               style=drop_style)
drop_residual = dcc.Dropdown(id='drop-3', options=[{"label": "Wykres reszt", "value": False},
                        {"label": "Wykres reszt linia", "value": True}], value=False,
                         placeholder='Wybierz rodzaj wykresu',className='dropdown',
                             style=drop_style)

dash.register_page(__name__, order=3)
layout = html.Div([
# tytul

    html.Div([
    dcc.Tabs(
        id='tabs-1',
        children=[
            dcc.Tab(label='XGBoost', value='tab-1',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='AdaBoost', value='tab-2',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='Lasy Losowe', value='tab-3',style = {'color':'white'},selected_style ={"background":'#035891'}),
            dcc.Tab(label='Drzewo Decyzyjne', value='tab-4',style = {'color':'white'},selected_style ={"background":'#035891'})
        ],
        value='tab-1',
    colors={
        "border":"#242424", #obwodka
        "background":'#242424', #tlo
        'primary':'#035891' #jesli wybrane

    },
),
    html.Div(id='div-1')
])])


@callback(
    Output('div-1', 'children'),
    [Input('tabs-1', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
        # pierwsza linia cards
        html.Div([

                html.Div([
                    html.H6(children='RMSE zbiór treningowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                        html.H6(children='20448',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='MODEL',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='XGBoost',
                            style={'textAlign': 'center',
                                   'color': '#035891'
                                            }),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='RMSE log zbiór testowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='0.138',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

        ],className='row flex display'),
        # druga linia cards
        html.Div([
                html.Div([
                    drop_ALE,
                    html.Br(),
                    dcc.Graph(id="graph-1", figure = figures[0])

                ], className='add_container six columns'),

                html.Div([
                    drop_importance,
                    html.Br(),
                    dcc.Graph(id="graph-2", figure=figures[1])
                ], className='add_container six columns'),


        ],className='row flex display'),
        # Residuals
        html.Div([
            drop_residual,
            html.Br(),
            dcc.Graph(id="graph-3", figure=figures[2])
            ], className='add_container twelve columns')
        ])
    elif tab == 'tab-2':
        return html.Div([
            # pierwsza linia cards
            html.Div([

                html.Div([
                    html.H6(children='RMSE zbiór treningowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='24738',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='MODEL',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='AdaBoost',
                            style={'textAlign': 'center',
                                   'color': '#035891'
                                   }),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='RMSE log zbiór testowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='0.159',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

            ], className='row flex display'),
            # druga linia cards
            html.Div([
                html.Div([
                    drop_ALE,
                    html.Br(),
                    dcc.Graph(id="graph-1", figure = figures[3])

                ], className='add_container six columns'),

                html.Div([
                    drop_importance,
                    html.Br(),
                    dcc.Graph(id="graph-2", figure=figures[4])
                ], className='add_container six columns'),

            ], className='row flex display'),
            # Residuals
            html.Div([
                drop_residual,
                html.Br(),
                dcc.Graph(id="graph-3", figure=figures[5])
            ], className='add_container twelve columns')
        ])
    elif tab == 'tab-3':
        return html.Div([
            # pierwsza linia cards
            html.Div([

                html.Div([
                    html.H6(children='RMSE zbiór treningowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='23111',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='MODEL',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='Lasy Losowe',
                            style={'textAlign': 'center',
                                   'color': '#035891'
                                   }),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='RMSE log zbiór testowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='0.151',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

            ], className='row flex display'),
            # druga linia cards
            html.Div([
                html.Div([
                    drop_ALE,
                    html.Br(),
                    dcc.Graph(id="graph-1", figure = figures[6])

                ], className='add_container six columns'),

                html.Div([
                    drop_importance,
                    html.Br(),
                    dcc.Graph(id="graph-2", figure = figures[7])
                ], className='add_container six columns'),

            ], className='row flex display'),
            # Residuals
            html.Div([
                drop_residual,
                html.Br(),
                dcc.Graph(id="graph-3", figure = figures[8])
            ], className='add_container twelve columns')
        ])
    elif tab == 'tab-4':
        return html.Div([
            # pierwsza linia cards
            html.Div([

                html.Div([
                    html.H6(children='RMSE zbiór treningowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='36521',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='MODEL',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='Drzewo Decyzyjne',
                            style={'textAlign': 'center',
                                   'color': '#035891'
                                   }),
                ], className='card_container four columns'),

                html.Div([
                    html.H6(children='RMSE log zbiór testowy',
                            style={'textAlign': 'center',
                                   'color': '#616161'}),
                    html.H6(children='0.238',
                            style={'textAlign': 'center',
                                   'color': 'white'}),
                ], className='card_container four columns'),

            ], className='row flex display'),
            # druga linia cards
            html.Div([
                html.Div([
                    drop_ALE,
                    html.Br(),
                    dcc.Graph(id="graph-1",figure = figures[9])

                ], className='add_container six columns'),

                html.Div([
                    drop_importance,
                    html.Br(),
                    dcc.Graph(id="graph-2", figure = figures[10])
                ], className='add_container six columns'),

            ], className='row flex display'),
            # Residuals
            html.Div([
                drop_residual,
                html.Br(),
                dcc.Graph(id="graph-3", figure=figures[11])
            ], className='add_container twelve columns')
        ])
@callback(
        [Output('graph-1', 'figure'),
         Output('graph-2', 'figure'),
         Output('graph-3', 'figure')
         ],
        [Input('drop-1', 'value'),
        Input('drop-2', 'value'),
        Input('drop-3', 'value'),
        Input('tabs-1', 'value')]
)
def update_graph(drop,drop_2,showline, tab):
    fig=dash.no_update
    fig_2=dash.no_update
    fig_3=dash.no_update
    if tab == "tab-1":
        label = xgb
        exp = exp_xgb
    elif tab == "tab-2":
        label = ada
        exp = exp_ada
    elif tab == "tab-3":
        label = rf
        exp = exp_rf
    elif tab == "tab-4":
        label = dt
        exp = exp_dt
    if drop is not None:
        explanation = exp.model_profile(variables=drop, N=None,
                                            label=label, type='accumulated')
        fig = explanation.plot(title="Wykres skumulowanych profili lokalnych", show=False, y_title="")
        fig.update_layout(yaxis=dict(showgrid=False, title='Cena Sprzedazy'),
                          xaxis=dict(showgrid=False, title="Wartość atrybutu"),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#616161', title_font={"size": 20})
        fig.layout.height = 400
        fig.layout.width = 550

    if drop_2 is not None:
        explanation = exp.model_parts(variables=drop_2,label=label)
        fig_2 = explanation.plot(title='Istotność zmiennych', digits=1, show=False)
        fig_2.update_layout(yaxis=dict(showgrid=False),
                      xaxis=dict(showgrid=False),paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)', font_color='#616161', title_font={"size": 20})
        fig_2.layout.height = 400
        fig_2.layout.width = 550
    if showline == False:
        size = 8  # marker size
        lsize = 0  # line_width
        range = None
        description = "Wykres_reszt"
    elif showline == True:
        size = 4  # marker size
        lsize = 4  # line_width
        range = [0, 50000]
        description = "Wykres_reszt_linia"
    if showline is not None:
        explanation = exp.model_diagnostics(label=label)
        fig_3 = explanation.plot(yvariable="abs_residuals", marker_size=size, line_width=lsize, show=False)
        fig_3.update_layout(
            title="Wartość przewidywana względem wartości resztowych", title_font_size=30,
            title_font_family="Lato, sans-serif",
            legend_title="Model uczenia maszynowego", legend_font_family="Lato, sans-serif",
            legend_title_font_size=15, legend_font_size=15,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Wartość absolutna reszty', showgrid=False, title_font={"size": 20},
                       title_font_family="Lato, sans-serif", range=range),
            xaxis=dict(title='Estymowana cena sprzedaży', showgrid=False, title_font={"size": 20},
                       title_font_family="Lato, sans-serif"),
            font_color='#616161', width=1400)
        fig_3.update_xaxes(automargin=True)
        fig_3.update_yaxes(automargin=True)
        fig_3.update_traces(marker_color='#4574a1')
        fig_3.layout.height = 400
        fig_3.layout.width = 1200
    return fig, fig_2, fig_3









