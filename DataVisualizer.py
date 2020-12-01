import json

import pandas as pd
import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc
import plotly.graph_objects as go
import calendar

from plotly.subplots import make_subplots
from DataHandler import DataHandler
from colour import Color


class DataVisualizer(object):

    def __init__(self, data_handler, app, mapbox_token, mapbox_style):
        self._data_handler = data_handler
        self._app = app

        self._default_web_title = "Rate My Professor Interactive Visualizer"
        self._default_web_credit = "Yi-Chen Liu, Jia-Wei Liang © 2020 Copyright held by the owner/author(s)."

        self._default_number_of_reserved = 8

        self._default_plain_fig = dict(
            data=[dict(x=0, y=0)],
            layout=dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, r=20, b=20, l=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
        )

        self._pca_clustering_df = {}
        self._pca_clustering_target = None
        self._pca_calc_trial_num = 1

        self._mapbox_token = mapbox_token
        self._mapbox_style = mapbox_style

        self.set_layout()
        self.callback()

    @classmethod
    def construct_from_csv(cls, path):
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        mapbox_info_file_path = "assets/mapbox_info.json"

        # with open(mapbox_info_file_path) as mapbox_info_file:
        #     mapbox_info_dict = json.load(mapbox_info_file)

        return DataVisualizer(
            data_handler=DataHandler.construct_from_csv(path),
            app=dash.Dash(__name__, external_stylesheets=external_stylesheets),
            mapbox_token=None,
            mapbox_style=None
        )

    def set_layout(self):
        self._app.title = self._default_web_title
        self._app.layout = html.Div(id="main", children=[
            html.Div(
                id="screen0",
                children=[
                    html.Div(
                        id="screen00",
                        children=[
                            dcc.Graph(id="map_rating_heat_map",
                                      figure=self._default_plain_fig,
                                      className="graph_style")
                        ]
                    ),

                    html.Div(
                        id="screen01",
                        children=[
                            html.Div(
                                id="screen01_top",
                                children=[
                                    dcc.Graph(id="bar_attributes_freq",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]),
                            html.Div(
                                id="screen01_bottom",
                                children=[
                                    html.H1(self._default_web_title)
                                ]
                            )

                        ]
                    )
                ]
            ),

            html.Div(
                id="screen1",
                children=[
                    html.Div(
                        id="screen10",
                        children=[
                            dcc.Dropdown(
                                id="student_school_list",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_school_list],
                                value=self._data_handler.get_school_list[0],
                                multi=False
                            ),

                            dcc.Dropdown(
                                id="student_department_list",
                                multi=False
                            ),

                            dcc.Dropdown(
                                id="student_designated_professor",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_professor_list],
                                value=self._data_handler.get_professor_list[0],
                                multi=False
                            ),

                            dcc.Dropdown(
                                id="student_top_willing_attributes",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_attribute_list],
                                multi=True
                            ),

                            dcc.Dropdown(
                                id="student_top_unwilling_attributes",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_attribute_list],
                                multi=True
                            )
                        ]
                    ),

                    html.Div(
                        id="screen11",
                        children=[
                            html.Div(
                                id="screen11_top",
                                children=[
                                    html.Div(
                                        id="screen11_top_overview",
                                        children=[
                                            html.Div(id="student_designated_professor_name"),
                                            html.Div(id="student_designated_professor_score"),
                                            html.Div(id="student_designated_professor_difficulty")
                                        ]
                                    ),
                                    html.Div(id="screen11_top_match_score"),
                                ]
                            ),

                            html.Div(
                                id="screen11_bottom",
                                children=[
                                    dcc.Graph(id="bar_matched_professors",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            )
                        ]
                    )
                ]
            ),

            html.Div(
                id="screen2",
                children=[
                    html.Div(
                        id="screen20",
                        children=[
                            html.Div(
                                id="screen200",
                                children=[
                                    dcc.Dropdown(
                                        id="prof_designated_professor",
                                        multi=False
                                    ),

                                    dcc.Dropdown(
                                        id="prof_top_willing_attributes",
                                        multi=True
                                    )
                                ]
                            ),
                            html.Div(
                                id="screen201",
                                children=[
                                    "This is NN Prediction!!!!!"
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id="screen21",
                        children=[
                            html.Div(
                                id="screen210",
                                children=[
                                    dcc.Graph(id="pca",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            ),

                            html.Div(
                                id="screen211",
                                children=[
                                    dcc.Graph(id="k-means",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            )
                        ]
                    )
                ]
            )
        ])

    def callback(self):

        self._app.callback(
            dash.dependencies.Output("student_department_list", "options"),
            dash.dependencies.Output("student_department_list", "value"),
            [dash.dependencies.Input("student_school_list", "value")]
        )(self._update_student_department_list)

        self._app.callback(
            dash.dependencies.Output("student_designated_professor", "options"),
            dash.dependencies.Output("student_designated_professor", "value"),
            [dash.dependencies.Input("student_school_list", "value"),
             dash.dependencies.Input("student_department_list", "value")]
        )(self._update_student_designated_professor_name)

        self._app.callback(
            dash.dependencies.Output("student_designated_professor_name", "children"),
            dash.dependencies.Output("student_designated_professor_score", "children"),
            dash.dependencies.Output("student_designated_professor_difficulty", "children"),
            [dash.dependencies.Input("student_school_list", "value"),
             dash.dependencies.Input("student_department_list", "value"),
             dash.dependencies.Input("student_designated_professor", "value")]
        )(self._update_student_designated_professor_overview)

    def _update_student_department_list(self, school_name):
        df = self._data_handler.get_data_frame_original
        department_list = df[df["school_name"] == school_name]["department_name"].unique()
        option_list = [{"label": element, "value": element} for element in department_list]
        value = department_list[0]

        return option_list, value

    def _update_student_designated_professor_name(self, school_name, department_name):
        df = self._data_handler.get_data_frame_original
        professor_list = df[(df["school_name"] == school_name) & (df["department_name"] == department_name)]["professor_name"].unique()
        option_list = [{"label": element, "value": element} for element in professor_list]
        value = professor_list[0]

        return option_list, value

    def _update_student_designated_professor_overview(self, school_name, department_name, professor_name):
        df = self._data_handler.get_data_frame_original
        df = df[(df["school_name"] == school_name) & (df["department_name"] == department_name)]
        df = df[["professor_name", "student_star", "student_difficult"]].groupby("professor_name").mean()

        return professor_name, df.loc[professor_name, "student_star"], df.loc[professor_name, "student_difficult"]

    @staticmethod
    def _get_color_scale(steps, c_from, c_to):
        return [color.hex for color in list(Color(c_from).range_to(Color(c_to), steps))]

    def _get_matched_score(self, professor_id, willing_list, unwilling_list):
        df = self._data_handler.get_data_frame_original

        return

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/RateMyProfessor_Sample data.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
