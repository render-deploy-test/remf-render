import dash
from dash import html, dcc


dash.register_page(__name__, path='/', order=0)

layout = html.Div(children=[
    html.H6('Autor Jakub Zawistowski', style={'textAlign': 'center','color':'#a9d4a7' }),
    html.Br(),
    html.H1('Streszczenie',style={'textAlign': 'center','color':'#eba134'}),
    dcc.Markdown('''###### Celem projektu jest predykcja ceny sprzedaży nieruchomości oraz określenie jakie cechy najbardziej oddziałują na jej końcową wartość. W tym celu wytrenowane zostały algorytmy typu black-box tj.: AdaBoost, XGBoost, Lasy Losowe, które zostały porównane z modelem typu white-box tj. Drzewem Decyzyjnym.
    
    
###### Przedstawione zostało kompleksowe rozwiązanie od przygotowania danych poprzez selekcję cech do analizy eksploracyjnej. Szczególny nacisk położono na zrozumienie predykcji dokonywanych przez porównywane modele - zarówno na poziomie lokalnym jak i globalnym. Wykorzystane zostały różne techniki wyjaśnienia modeli takie jak:

* ###### aproksymacja modelem zastępczym,
* ###### wartości Shapley’a,
* ###### zależności typu Break-Down, skumulowanych profili lokalnych.
 
 
###### Wyniki przedstawiono w interaktywnej aplikacji napisanej przy użyciu frameworka Dash.''',
                 style={'textAlign': 'left','color': 'white'}),

])

