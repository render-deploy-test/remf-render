from sklearn import tree
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from category_encoders import TargetEncoder
import scipy
import pandas as pd
import plotly.graph_objects as go
import dalex as dx
import graphviz
import pylab
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NumericScaler(BaseEstimator, TransformerMixin):
    """
    Klasa NumericScaler zawiera 2 metody fit oraz transform. Przeznaczona do transformacji atrybutów numerycznych.
    """
    def __init__(self):
        self.skew_num = None
        self.other_num = None
        self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    def fit(self, X, y=None):
        """
        Podział zbioru treningowego na 2 podzbiory w zależności od wartości skośności kolumn.
        :param X: dane treningowe - typu dataframe
        :param y: parametr zostawiony dla kompatybilności
        :return:self
        Nazwy kolumn o skośności powyżej 1 przypisywane sa zmiennej self.skew_num.
        Nazwy kolumn o skośności poniżej = 1 przypisywane sa do zmiennej self.other_num
        """
        X = pd.DataFrame(X)
        num = X.select_dtypes(include=self.numerics)
        skew_num = num.apply(lambda x: scipy.stats.skew(x))
        skew_num = skew_num[skew_num > 1]
        self.skew_num = skew_num.index
        self.other_num = num.drop(skew_num.index,axis=1).columns
        return self
    def transform(self, X, y=None, **fit_params):
        """
        Transformacja:
        Kolumny o skośności powyżej 1 poddane zostaan transformacji log1p
        Kolumny o skośności poniżej/równe 1 przeskalowane zostaną przy użyciu RobustScaler().

        :param X: dane treningowe - typu dataframe (n_samples, n_features)
        :param y: parametr pozostawiony dla kompatybilności
        :param fit_params:
        :return:DataFrame zawierajacy przekształcone zmienne numeryczne.
        """
        X = pd.DataFrame(X)
        z = X.shape
        X = X.select_dtypes(include=self.numerics)
        X[self.skew_num] = X[self.skew_num].apply(lambda x: np.log1p(x))
        X[self.other_num] = RobustScaler().fit_transform(X[self.other_num])
        if z != X.shape:
            raise ValueError("Rozmiar DataFrame'u po transofrmacji logarytmicznej jest inny od pierwotnego")
        return pd.DataFrame(X)

class CatScaler(BaseEstimator, TransformerMixin):
    """
    Klasa CatScaler zawiera 2 metody fit oraz transform. Przeznaczona do transformacji atrybutów kategorycznych.
    """
    def __init__(self):
        self.categorical_features = None
        self.categorical_features_high = None
        self.transformer = None
        self.transformed = None
    def fit(self, X, y):
        """
        Podział zbioru treningowego na 2 podzbiory w zależności od wartości wartości unikalnych kolumn.
        Zmienne kategoryczne o liczbie wartości unikalnych poniżej 6 są kodowane metodą binarną,
         pozostałe są przekształcone do postaci numerycznej metodą TargetEncoder.
        :param X: zmienne objaśniające
        :param y: zmienna objasniana
        :return: self

        Note : Ze względu na TargetEncoder do y nie można podstawić none (potrzebne do fitowania TE).
        Indexy resetowane ze względu na podział zbioru w gridsearchu oraz potrzebne y do fitowania TargetEncodera.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.categorical_features = [col for col in X.columns if X[col].nunique() < 6]
        self.categorical_features_high = list(set(X.columns) - set(self.categorical_features))
        self.transformer = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_features),
            (TargetEncoder(cols=self.categorical_features_high, return_df=True), self.categorical_features_high),
            remainder="passthrough")
        X_merge = pd.concat([X, y.to_frame()], axis=1, ignore_index=True)
        self.fitted = self.transformer.fit(X_merge.iloc[:,:-1],X_merge.iloc[:,-1:])
        return self
    def transform(self, X, y=None, **fit_params):
        """
        Transformacja:
        Zmienne kategoryczne o liczbie wartości unikalnych poniżej 6 są kodowane metodą binarną,
         pozostałe są przekształcone do postaci numerycznej metodą TargetEncoder.

        :param X: zmienne objaśniające
        :param y: zmienna objasniana
        :param fit_params:
        :return: DataFrame zawierajacy przekształcone zmienne kategoryczne.
        """
        X = pd.DataFrame(X)
        transformed = self.fitted.transform(X)
        return transformed

class MakeModel():
    """
    Tworzenie modeli uczenia maszynowego
    :param models: lista przechowujaca modele uczenia maszynowego
    :param names: lista przechowujaca nazwy modeli uczenia maszynowego
    :param grid: lista przechowujaca najlepsze estymatory dobrane przy pomocy metody gridsearch
    :param kf: parametry walidacji krzyżowej
    :param explainers: lista przechowujaca explainery wytrenowych modeli uczenia maszynowego
    """
    def __init__(self):
        self.models = []
        self.names = []
        self.grid = []
        self.kf = KFold(n_splits=10, random_state=42, shuffle=True)
        self.explainers = []
    def add_model(self, name, model): # name - string
        """
        Dodaj model uczenia maszynowego.
        :param name: nazwa modelu typ string
        :param model: zastosowany model
        :return: self
        """
        log = TransformedTargetRegressor(model, func = np.log, inverse_func=np.exp)
        pipe = Pipeline(steps=[('preprocessor', self.preprocessor),('regressor', log)])
        self.models.append(pipe)
        self.names.append(name)
        self.grid.append(pipe)
        return self
    def add_preprocessor(self, preprocessor): # sprawdzic typ czy pipeline
        """
        Dodaj definicję preprocessora wykorzystywana do trenowania modeli.
        :param preprocessor: zdefiniowany preprocessor np.ColumnTransformer
        :return: self
        """
        self.preprocessor = preprocessor
        return self

    def gridsearch(self,name, X, Y, parameters): # sprawdzic czy name nalezy do self.names
        """
        Dobór hiperparametrów metoda Gridsearch dla wybranego modelu po zdefiniowanych parametrach.
        :param name: zdefiniowana w add_model nazwa modelu uczenia maszynowego
        :param X: dane treningowe - typu dataframe (n_samples, n_features)
        :param Y: zmienna objaśniana
        :param parameters: parametry modelu uczenia maszynowego typu dict
        :return: self
        """
        cls = self.models[self.names.index(name)]
        model = GridSearchCV(cls, parameters, cv=self.kf, scoring='neg_mean_squared_error', n_jobs= -1, verbose=1,
                             error_score='raise')
        model.fit(X, Y)
        print("Najlepszy estymator: ", model.best_estimator_,
              "\nNajlepszy wynik: ", np.sqrt(abs(model.best_score_)),
              "\nNajlepszy parametr: ", model.best_params_, sep="\n")
        self.grid[self.names.index(name)] = model # bylo best estimator
        return self
    def print_results(self):
        """
        Wypisz wyniki modeli osiagniete na zbiorze treningowym.
        :return: self
        """
        for i in self.names:
            print("Najlepszy wynik {0}: ".format(i), np.sqrt(abs(self.grid[self.names.index(i)].best_score_)))
        return self
    def plot_results(self):
        """
        Zwraca wykres przedstawiajacy wyniki zastosowanych modeli uczenia maszynowego na zbiorze treningowym.
        :return: figure
        """
        RMSE = []
        for i in self.grid:
            RMSE.append(str(round(np.sqrt(abs(i.best_score_)),2)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.names, y=RMSE, name='RMSE', text=list(map(str , RMSE)), line=dict(color='#3440eb', width=4)))
        fig.update_layout(
            title="Zestawienie wyników modeli uczenia maszynowego",
            xaxis_title="Nazwa algorytmu", yaxis_range=[18000, 40000],
            yaxis_title="RMSE", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black', size=15), title_x=0.5, title_font_size=20, width=1300
        )
        fig.update_xaxes(showgrid=False, ticksuffix=' ', automargin=True)
        fig.update_yaxes(showgrid=False, ticksuffix=' ', automargin=True)
        fig.update_traces(marker=dict(size=15, color='black'), mode="lines+markers+text", textposition="top center")
        return fig.show()
    def predict(self,path, X_test):
        """
        Zapisz wyniki predykcji modeli do folderu submissions
        :param path: path do sample submission.csv tj. "Ames_Data_Set/sample_submission.csv"
        :param X_test: zmienne objaśniajace
        :return: None
        """
        for i in self.names:
            submission = pd.read_csv(path)
            submission.iloc[:, 1] = np.floor(self.grid[self.names.index(i)].predict(X_test))
            submission.to_csv("submissions/submission_regression_{0}.csv".format(i), index=False)
    def explain(self, X_train,Y_train):
        """
        Stwórz explainery (pakiet Dalex) dla wszystkich trenowanych modeli uczenia maszynowego. W celu wyjaśnienia
        działania modeli dokonywany jest podział na zbiór treningowy i testowy w proporcji 0.75:0.25.
        :param X_train: zmienne objaśniajace
        :param Y_train: zmienna objaśniana
        :return: self
        """
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
        for model in self.grid:
            model.fit(x_train, y_train)
            self.explainers.append(dx.Explainer(model, x_test, y_test))
        return self
    def plot_importance(self, vars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars',
                                 'GarageArea', 'YearRemodAdd', 'Fireplaces', 'LotArea','BsmtFinSF1']):
        """
        Metoda zwraca wykres istotności zmiennych dla wszystkich trenowanych modeli uczenia maszynowego.
        Wygenerowany wykres zapisywany jest do folderu images.
        :param vars: definicja zmiennych objasniajacych, które będa widoczne na wykresie
        :return: figure
        """
        objects = []
        for i in self.explainers:
            objects.append(i.model_parts(label=self.names[self.explainers.index(i)]))
        objects.pop(0)
        explanation = self.explainers[0].model_parts(label=self.names[0], variables=vars)
        fig = explanation.plot(objects=objects,title='Istotność zmiennych',digits=1,show=False)
        fig.update_layout(yaxis=dict(showgrid=False),
                      xaxis=dict(showgrid=False, title="Istotność cechy"))
        fig.write_image("images/Feature_imp.png")
        return fig.show()
    def plot_pdp(self, numeric_vars=['GrLivArea','OverallQual','TotalBsmtSF','GarageArea']):
        objects = []
        for i in self.explainers:
            objects.append(i.model_profile(variables=numeric_vars, N=None, label=self.names[self.explainers.index(i)]))
        objects.pop(0)
        explanation = self.explainers[0].model_profile(variables=numeric_vars, N=None,
                                                       label=self.names[self.explainers.index(0)])
        fig = explanation.plot(objects=objects, title = "", show=False)
        fig.write_image("images/Partial_Dependence_Profile.png")
        return fig.show()
    def plot_acc(self, numeric_vars=['GrLivArea','OverallQual','TotalBsmtSF','GarageArea']):
        """
        Metoda zwraca wykres skumulowanych profili lokalnych dla zastosowanych modeli uczenia maszynowego.
        Wygenerowany wykres zapisywany jest do folderu images.
        :param numeric_vars: zmienne numeryczne, których wykresy zostaną przedstawione
        :return: figure
        """
        objects = []
        for i in self.explainers:
            objects.append(i.model_profile(variables=numeric_vars, N=None, label=self.names[self.explainers.index(i)],
                                           type='accumulated'))
        objects.pop(0)
        explanation = self.explainers[0].model_profile(variables=numeric_vars, N=None,
                                                       label=self.names[0],type='accumulated') #tutaj zmiana była na 0
        fig = explanation.plot(objects=objects, title=" ", show=False, y_title='Predykcja')
        fig.update_layout(legend_title="Zastosowany model uczenia maszynowego",title_x=0.5,
                                  legend_font_family="Lato, sans-serif",width=1500, height = 900,
                    yaxis=dict(automargin=True),
                    xaxis=dict(title='Wartości cechy', automargin=True,ticklabelposition = 'inside bottom'),)
        fig.write_image("images/Accumulated.png")
        return fig.show()
    def plot_residuals(self, showline=False):
        """
        Metoda zwraca wykres wartości resztowych dla zastosowanych modeli uczenia maszynowego.
        Wygenerowany wykres zapisywany jest do folderu images.
        :param showline: definicja czy na wykresie ma być widoczna linia wygładzajaca
        :return: figure
        """
        if showline == False: # mozna dodac test, ze showline moze byc tylko True i False
            size = 8 #marker size
            lsize=0 # line_width
            range = None
            description = "Wykres_reszt"
        elif showline == True:
            size = 4 #marker size
            lsize = 4 # line_width
            range = [0,50000]
            description = "Wykres_reszt_linia"
        objects = []
        for i in self.explainers:
            objects.append(i.model_diagnostics(label=self.names[self.explainers.index(i)]))
        objects.pop(0)
        explanation = self.explainers[0].model_diagnostics(label=self.names[0])
        fig = explanation.plot(objects=objects, yvariable="abs_residuals",marker_size=size, line_width=lsize, show=False)
        fig.update_layout(
            title="Wartość przewidywana względem wartości resztowych", title_font_size=30,
            title_font_family="Lato, sans-serif",
            legend_title="Zastosowany model uczenia maszynowego", legend_font_family="Lato, sans-serif",
            legend_title_font_size=15,legend_font_size=15,
            yaxis=dict(title='Wartość absolutna reszty', showgrid=False, title_font={"size": 20},
                       title_font_family="Lato, sans-serif", range = range),
            xaxis=dict(title='Estymowana cena sprzedaży', showgrid=False, title_font={"size": 20},
                       title_font_family="Lato, sans-serif"),
            font_color='black', width=1400)
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        fig.write_image("images/{0}.png".format(description))
        return fig.show()

    def plot_breakdown(self,X, nr_instance, name, order = np.array(
            ['OverallCond', "YearRemodAdd", 'TotRmsAbvGrd', 'BsmtFinSF1', 'LotArea', 'Fireplaces', 'WoodDeckSF',
             'TotalBsmtSF', 'BsmtFullBath', 'GarageArea', 'OverallQual', "YearBuilt", 'GrLivArea'])): # nr_instance - numer analizowanej instancji, X - zbiór treningowy, name - nazwa modelu string
        to_pred = X.iloc[[nr_instance]]
        bd_pred = self.explainers[self.names.index(name)].predict_parts(to_pred[X.columns], type='break_down',
                                                                        order=order, label=name)
        fig = bd_pred.plot(max_vars=90, title='Dekompozycja predykcji - top {0} atrybutów'.format(len(order)),
                           digits=1, show=False)
        update_layout = fig.update_layout(title_font_size=30,
                                          title_font_family="Lato, sans-serif",
                                          legend_font_family="Lato, sans-serif", title_x=0.5,
                                          yaxis=dict(title='Nazwa atrybutu', showgrid=True, title_font={"size": 20},
                                                     title_font_family="Lato, sans-serif"),
                                          xaxis=dict(showgrid=False, title_font={"size": 20},
                                                     title_font_family="Lato, sans-serif"), font_color='black',
                                          width=1400, )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        fig.write_image("images/Break_down.png")
        return fig.show()
    def shap(self,X, nr_instance, name):
        """
        Metoda zwraca wykres wartości shapleya dla wybranego modelu uczenia maszynowego.
        Wygenerowany wykres zapisywany jest do folderu images.
        :param X: zmienne objaśniające
        :param nr_instance: numer instancji, która będzie analizowana
        :param name: nazwa modelu uczenia maszynowego
        :return: figure
        """
        to_pred = X.iloc[[nr_instance]]
        bd_pred_shap = self.explainers[self.names.index(name)].predict_parts(to_pred[X.columns], type='shap', label=name)
        fig = bd_pred_shap.plot(max_vars=20, digits=1, show=False)
        fig.update_layout(title_font_size=30, title='Wartości Shapleya',
                          title_font_family="Lato, sans-serif",
                          legend_font_family="Lato, sans-serif", title_x=0.5,
                          yaxis=dict(title='Nazwa atrybutu', showgrid=True, title_font={"size": 20},
                                     title_font_family="Lato, sans-serif"),
                          xaxis=dict(showgrid=False, title_font={"size": 20},
                                     title_font_family="Lato, sans-serif"), font_color='black', width=1400, )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        fig.write_image("images/Shap_values.png")
        return fig.show()
    def tree_surrogate(self, name, max_depth =3, max_vars=10): # do sprawdzenia
        #exp = self.explainers[self.names.index(name)]
        exp = self.explainers[0]
        surrogate_model = exp.model_surrogate(type='tree', max_depth=max_depth,
                                                                                  max_vars=max_vars)
        dot_data = tree.export_graphviz(surrogate_model, out_file=None, rounded=True, precision=1,
                                        class_names='SalePrice', feature_names=surrogate_model.feature_names,
                                        filled=True, proportion=True)
        graph = graphviz.Source(dot_data, format="png")
        graph.render(filename='g1.dot')
        pylab.savefig('filename.png')
        return surrogate_model.plot(figsize=(20, 15), fontsize=12, filled=True)
    def save_explainers(self,path):
        """
        Zapisz wszystkie Explainery modeli do wskazanej lokalizacji
        :param path: lokalizacja, w której maja zostać zapisane Explainery
        :return: None
        """
        for i in self.names:
            with open(path+"_{0}.pkl".format(i), 'wb') as fd:
                 self.explainers[self.names.index(i)].dump(fd)
        return None
    def load_explainers(self,path):
        """
        Wczytaj zapisane Explainery.
        Uwaga - metodę należy uruchamiać na instancji bez zdefiniowanych Explainerów/modeli ("czysta instancja klasy")
        :param path: lista ścieżek, w których zapisane sa Explainery. Ścieżki musza byc podane w formacie string.
        :return: None
        """
        for i in path:
            with open(i, 'rb') as fd:
                dx.Explainer.load(fd)
        return None

