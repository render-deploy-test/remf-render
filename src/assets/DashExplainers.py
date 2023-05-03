import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import dalex as dx
from plotly.io import to_json, read_json



class DashExplainers:
    def __init__(self, train_file, test_file):
        self.df = None
        self.df_test = None
        self.train_file = train_file
        self.test_file = test_file
        self.explainers = {}
        self.importance = {}
        self.ale = {}
        self.residuals = {}

    def load_data(self):
        """
        Wczytaj dane
        :return: dataframe, dataframe - zbiór treningowy i testowy
        """
        df = pd.read_csv(self.train_file, header=0)
        df_test = pd.read_csv(self.test_file, header=0)
        self.df, self.df_test = df, df_test
        self.num_cols = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        return df, df_test

    def load_explainers(self, path, name=str()):
        """
        Funkcja wczytujaca zapisany Explainer (pakiet Dalex). W zastosowanym rozwiazaniu kolejność wczytania Explainerów
        ma znaczenie.
        :param path: ścieżka, w której został zapisany Explainer
        :param name: nazwa Explainer'a, typu string
        :return: Explainer, name
        """
        with open(path, 'rb') as fd:
            self.explainers[name] = dx.Explainer.load(fd)
        return self.explainers[name]
    def plot_importance(self):
        """
        Funkcja, ktora tworzy i zapisuje wykresy istotnosci zmiennych wszystkich wczytanych Explainerów do
        słownika self.importance
        :return: None
        """
        for key,value in self.explainers.items():
            explanation = value
            explanation = explanation.model_parts(
            variables=['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars',
                       'GarageArea', 'YearRemodAdd', 'Fireplaces', 'LotArea', 'BsmtFinSF1'], label=key)
            fig = explanation.plot(title='Istotność zmiennych', digits=1, show=False)
            fig.layout.height = 400
            fig.layout.width = 550
            fig.update_layout(yaxis=dict(showgrid=False),
                           xaxis=dict(showgrid=False),
                              paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#616161',
                              title_font={"size": 20})
            #fig.update_traces(marker_color='red')
            self.importance[key] = fig
        return None

    def plot_ALE(self):
        """
        Funkcja, ktora tworzy i zapisuje wykresy skumulowanych profili lokalnych wszystkich wczytanych Explainerów
        do słownika self.ale
        :return: None
        """
        for key, value in self.explainers.items():
            explanation = value
            explanation = explanation.model_profile(variables='GrLivArea', N=None,
                                        label=key, type='accumulated')
            fig = explanation.plot(title="Wykres skumulowanych profili lokalnych", show=False, y_title="")
            fig.layout.height = 400
            fig.layout.width = 550
            fig.update_layout(yaxis=dict(showgrid=False, title = 'Cena Sprzedazy'),
                              xaxis=dict(showgrid=False, title="Wartość atrybutu"),
                              paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#616161', title_font={"size": 20})
            self.ale[key] = fig
        return None

    def plot_residuals(self):
        """
        Funkcja, ktora tworzy i zapisuje wykresy wartosci resztowych wszystkich wczytanych Explainerów
        do słownika self.residuals
        :return: None
        """
        size = 8  # marker size
        lsize = 0  # line_width
        range = None
        description = "Wykres_reszt"
        for key, value in self.explainers.items():
            exp = value
            explanation = exp.model_diagnostics(label=key)
            fig = explanation.plot(yvariable="abs_residuals", marker_size=size, line_width=lsize, show=False)
            fig.update_layout(
                    title="Wartość przewidywana względem wartości resztowych", title_font_size=30,
                    title_font_family="Lato, sans-serif",
                    legend_title="Model uczenia maszynowego", legend_font_family="Lato, sans-serif",
                    legend_title_font_size=15, legend_font_size=15, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title='Wartość absolutna reszty', showgrid=False, title_font={"size": 20},
                               title_font_family="Lato, sans-serif", range=range),
                    xaxis=dict(title='Estymowana cena sprzedaży', showgrid=False, title_font={"size": 20},
                               title_font_family="Lato, sans-serif"),
                    font_color='#616161', width=1400)
            fig.update_xaxes(automargin=True)
            fig.update_yaxes(automargin=True)
            fig.update_traces(marker_color='#4574a1')
            fig.layout.height = 400
            fig.layout.width = 1200
            self.residuals[key] = fig
        return None
    def save_plots(self):
        """
        Zapisz wykresy istotności i skumulowanych profili lokalnych w folderze assets
        :return:None
        """
        for key, value in self.importance.items():
            with open("assets/importance_{0}.json".format(key), 'w') as f:
                f.write(to_json(value))
        for key, value in self.ale.items():
            with open('assets/ale_{0}.json'.format(key), 'w') as f:
                f.write(to_json(value))
        for key, value in self.residuals.items():
            with open("assets/residuals_{0}.json".format(key), 'w') as f:
                f.write(to_json(value))
        return None
    def load_plots(self,names = []):
        """
        Wczytaj wykresy wartości skumulowanych profili lokalnych, istotności zmiennych i wartości resztowych z plików
        json (zapisanych w folderze assets) i zwróć je zapisane w tabeli
        :return: list - tabela zawierająca zapisane wykresy typu figure
        """
        output = []
        for key in names:
            file = "assets/ale_{0}.json".format(key)
            output.append(read_json(file, output_type='Figure', skip_invalid=False, engine=None))
            file = "assets/importance_{0}.json".format(key)
            output.append(read_json(file, output_type='Figure', skip_invalid=False, engine=None))
            file = "assets/residuals_{0}.json".format(key)
            output.append(read_json(file, output_type='Figure', skip_invalid=False, engine=None))
        return output


