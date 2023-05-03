import dash
from dash import html, dcc,callback, Input, Output
import dill as pickle
import pandas as pd

with open('../src/assets/xgb.sav', 'rb') as file:
    model = pickle.load(file)
df = pd.read_csv('../src/assets/train.csv')
df_test = pd.read_csv('../src/assets/test.csv')
Y = df.SalePrice.copy()
X = df.drop(['SalePrice'], axis=1).copy()
#df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
numerical_features_im_with_ln=['GarageCars','OverallCond','OverallQual','YearBuilt','GrLivArea', 'TotalBsmtSF','LotArea']
categorical_features_im_high = ['Neighborhood']
drop_style = {'background-color': '#cfa527', 'textAlign': 'center', 'margin': 'auto', 'color':'black'}
features = ['Neighborhood','GarageCars','OverallCond','OverallQual','YearBuilt','GrLivArea', 'TotalBsmtSF','LotArea']
dash.register_page(__name__, order=4)
layout = html.Div([
    html.Div([
        html.Div(id='div-7'),
        html.Div(id='div-5'),
    ]),
    html.Div([
        html.Div([
            html.H6(children='Podaj dzielnicę',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            html.Br(),
            dcc.Dropdown(
                id='dropdown-2',
                options=[{'label': i, 'value': i} for i in df.Neighborhood.unique()],
                className='dropdown',
                placeholder='Podaj wartość',
                style=drop_style
            )
        ], className='card_container_2 three columns'),

        html.Div([
            html.H6(children='Liczba samochodów w garażu',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            dcc.Dropdown(
                id='dropdown-3',
                options=[{'label': i, 'value': i} for i in range(1, 6)],
                className='dropdown',
                placeholder='Podaj wartość',
                style=drop_style
            )
        ], className='card_container_2 three columns'),

        html.Div([
            html.H6(children='Ogólny stan nieruchomości',
                    style={'textAlign': 'center',
                           'color': '#616161'}),
            dcc.Dropdown(
                id='dropdown-4',
                options=[{'label': i, 'value': i} for i in range(1, 11)],
                className='dropdown',
                placeholder='Podaj wartość',
                style=drop_style
            )
        ], className='card_container_2 three columns'),

        html.Div([
            html.H6(children='Ogólny stan wykończenia domu', style={'textAlign': 'center',
                                                                    'color': '#616161'}),
            dcc.Dropdown(
                id='dropdown-1',
                options=[{'label': i, 'value': i} for i in range(1, 11)],
                className='dropdown',
                placeholder='Podaj wartość',
                style=drop_style
            )

        ], className='card_container_2 three columns'),
        html.Br(),
        html.Div([
            html.H6('Podaj rok budowy domu',style={'textAlign': 'center',
                           'color': 'white'}),
            dcc.Slider(
                id='slider-1',
                min=df.YearBuilt.min(),
                max=df.YearBuilt.max(),
                step=1,
                marks={i: str(i) for i in range(int(df.YearBuilt.min()), int(df.YearBuilt.max())+1, 10)},
                tooltip={'placement': 'bottom'}
            )
        ]),
        html.Br(),
        html.Div([
            html.H6('Powierzchnia domu [stopy kwadratowe]',style={'textAlign': 'center',
                           'color': 'white'}),
            dcc.Slider(
                id='slider-2',
                min=0,
                max=df.GrLivArea.max(),
                step=1,
                marks={i: str(i) for i in range(0, df.GrLivArea.max()+1, 1000)},
                tooltip={'placement': 'bottom'}
            )
        ]),
        html.Br(),
        html.Div([
            html.H6('Powierzchnia piwnicy [stopy kwadratowe]',style={'textAlign': 'center',
                           'color': 'white'}),
            dcc.Slider(
                id='slider-3',
                min=df.TotalBsmtSF.min(),
                max=df.TotalBsmtSF.max(),
                step=1,
                marks={i: str(i) for i in range(int(df.TotalBsmtSF.min()), int(df.TotalBsmtSF.max()) + 1, 1000)},
                tooltip={'placement': 'bottom'}
            )
        ]),
        html.Br(),
        html.Div([
            html.H6('Powierzchnia działki [stopy kwadratowe]',style={'textAlign': 'center',
                           'color': 'white'}),
            dcc.Slider(
                id='slider-4',
                min=1000,
                max=df.LotArea.max(),
                step=1,
                marks={i: str(i) for i in range(1000, int(df.LotArea.max()) + 1, 25000)},
                tooltip={'placement': 'bottom'}
            )
        ]),
    ], className='row flex display'),

])

@callback(
    Output('div-5', 'children'),
    [Input('dropdown-2', 'value'),
    Input('dropdown-3', 'value'),
    Input('dropdown-4', 'value'),
    Input('dropdown-1', 'value'),
    Input('slider-1', 'value'),
    Input('slider-2', 'value'),
    Input('slider-3', 'value'),
    Input('slider-4', 'value')
     ]
)
def display_parameters(value_1, value_2,value_3, value_4,value_5, value_6,value_7, value_8):
    if value_1 and value_2 and value_3 and value_4 and value_5 and value_6 and value_7 and value_8:
        return html.Div([
            html.H5('Wybrane parametry',style={'textAlign': 'center',
                           'color': '#035891'}),
            html.H6("Dzielnica {dzielnica}".format(dzielnica=value_1),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Liczba samochodów w garażu {liczba}".format(liczba=value_2),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Ogólny stan nieruchomości {jakosc}".format(jakosc = value_3),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Ogólny stan wykonczenia domu {jakosc}".format(jakosc = value_4),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Rok budowy domu {rok}".format(rok=value_5),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Powierzchnia domu {powierzchnia} stóp kwadratowych".format(powierzchnia=value_6),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Powierzchnia piwnicy {piwnica} stóp kwadratowych".format(piwnica=value_7),style={'textAlign': 'center',
                           'color': 'white'}),
            html.H6("Powierzchnia działki {powierzchnia} stóp kwadratowych".format(powierzchnia=value_8),style={'textAlign': 'center',
                           'color': 'white'}),
        ], className='card_container twelve columns')
@callback(
    Output('div-7', 'children'),
    [Input('dropdown-2', 'value'),
    Input('dropdown-3', 'value'),
    Input('dropdown-4', 'value'),
    Input('dropdown-1', 'value'),
    Input('slider-1', 'value'),
    Input('slider-2', 'value'),
    Input('slider-3', 'value'),
    Input('slider-4', 'value')
     ]
)
def wycena(value_1, value_2,value_3, value_4,value_5, value_6,value_7, value_8):
    if value_1 and value_2 and value_3 and value_4 and value_5 and value_6 and value_7 and value_8:
        dataframe = pd.DataFrame(
            data={'Neighborhood': [value_1],
                    'GarageCars':[value_2],
                  'OverallCond':[value_3],
                  'OverallQual': [value_4],
                  'YearBuilt': [value_5],
                  'GrLivArea': [value_6],
                  'TotalBsmtSF': [value_7],
                  'LotArea': [value_8],
                  }
        )
        wynik = model.predict(dataframe)
        wynik = wynik[0]
        return html.Div([
            html.H6("Estymowana cena sprzedaży $ {cena}".format(cena=int(wynik)),
                    style={'textAlign': 'center',
                           'color': 'white'}),

        ], className='card_container twelve columns')
    else:
        return html.Div([
            html.H4(children='W celu otrzymania wyceny podaj wszystkie parametry nieruchomości',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.H4(children='Zastosowany model ML - XGBoost trenowany na podstawie 8 atrybutów.',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.Br(),
            html.H4(children='Wersja demonstracyjna. Ze względu na znacznie mniejsza ilość atrybutów przeznaczonych do '
                             'trenowania modelu - jego predykcje sa mniej dokładne niz modeli wytrenowanych w'
                             ' ramach projektu.',
                    style={'textAlign': 'center',
                           'color': 'white'})
        ], className='card_container twelve columns')