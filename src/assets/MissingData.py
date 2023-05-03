import pandas as pd
import plotly.graph_objects as go

class MissingData:
    def __init__(self, df):
        self.df = df

    def return_miss(self):
        return self.df.columns[self.df.isnull().sum() > 0]

    def sum_miss(self, columns):
        return self.df[columns].isnull().sum().sort_values(ascending=False)

    def miss_data_hist(self, title='Brakujące dane'):
        z = self.return_miss()
        miss_data = go.Bar(
            x=self.sum_miss(z).index,
            y=self.sum_miss(z),
            text=self.sum_miss(z)
        )
        missing_data = [miss_data]
        layout = go.Layout(
            title=title,
            yaxis=dict(title='Liczba', showgrid=False, title_font={"size": 20}),
            xaxis=dict(title='Nazwa zmiennej', showgrid=False, title_font={"size": 20}),
            font_color='black'
        )
        fig = go.Figure(data=missing_data, layout=layout)
        fig.update_traces(
            marker_color=["tomato", "darkblue", "gold", "lightblue", "greenyellow", "lightslategrey", "antiquewhite",
                          "aqua", "darkred", "lightsteelblue", "beige", "bisque", "antiquewhite", "blanchedalmond",
                          "darkgray", "darkslategrey", "darkslateblue", "darkred", "darkorchid"],
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, opacity=0.6, texttemplate='%{text:.4s}', textposition='auto', constraintext='both')
        fig.update_layout(title_font={'size': 35}, title_x=0.5,
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        return fig.show()

    def fill_mode(self,columns=[]):
        for i in columns:
            self.df[i] = self.df[i].transform(lambda x: x.fillna(x.mode()[0]))
        return self.df

    def fill_none(self, columns=[]):
        self.df[columns] = self.df[columns].fillna("None")
        return self.df

    def fill_median(self, columns=[]):
        for i in columns:
            self.df[i]=self.df[i].fillna(self.df[i].median())
        return self.df

    def change_type(self,columns=[], type_to_change='str'):
        self.df[columns] = self.df[columns].astype(type_to_change)
        return self.df

    def delete_columns(self, columns=[]):
        self.df = self.df.drop(columns, axis=1)
        return self.df

