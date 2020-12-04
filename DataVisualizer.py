import json
import collections

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

        with open(mapbox_info_file_path) as mapbox_info_file:
            mapbox_info_dict = json.load(mapbox_info_file)

        return DataVisualizer(
            data_handler=DataHandler.construct_from_csv(path),
            app=dash.Dash(__name__, external_stylesheets=external_stylesheets),
            mapbox_token=mapbox_info_dict["mapbox_token"],
            mapbox_style=mapbox_info_dict["mapbox_style"]["UCDavis_289H_Project2_Dark"]
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
                            html.H3(children="Select Student's School"),
                            dcc.Dropdown(
                                id="student_school_list",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_school_list],
                                value=self._data_handler.get_school_list[0],
                                multi=False
                            ),

                            html.H3(children="Select Student's Department"),
                            dcc.Dropdown(
                                id="student_department_list",
                                multi=False
                            ),

                            html.H3(children="Select Designated Professor"),
                            dcc.Dropdown(
                                id="student_designated_professor",
                                multi=False
                            ),

                            html.H3(children="Select Willing Attributes Students Care Most"),
                            dcc.Dropdown(
                                id="student_top_willing_attributes",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_all_tag_list],
                                value=self._data_handler.get_all_tag_list[:5],
                                multi=True
                            ),

                            html.H3(children="Select Unwilling Attributes Students Aware Most"),
                            dcc.Dropdown(
                                id="student_top_unwilling_attributes",
                                options=[{"label": element, "value": element} for element in
                                         self._data_handler.get_all_tag_list],
                                value=self._data_handler.get_all_tag_list[6:8],
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
                                        id="prof_school_list",
                                        options=[{"label": element, "value": element} for element in
                                                 self._data_handler.get_school_list],
                                        value=self._data_handler.get_school_list[0],
                                        multi=False
                                    ),

                                    dcc.Dropdown(
                                        id="prof_department_list",
                                        multi=False
                                    ),

                                    dcc.Dropdown(
                                        id="prof_designated_professor",
                                        multi=False
                                    ),

                                    dcc.Dropdown(
                                        id="prof_top_willing_attributes",
                                        multi=True
                                    ),

                                    html.H3(children="PCA/ K-Means Feature Columns Picker"),
                                    dcc.Dropdown(
                                        id="ml_feature_cols_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_numeric_columns],
                                        value=self._data_handler.get_numeric_columns,
                                        multi=True
                                    ),

                                    html.Div(id="side_bar_bottom1_parameters", children=[
                                        html.Div(id="ml_num_of_pc_frame", children=[
                                            html.H4(children="Principle Columns"),
                                            dcc.Input(
                                                id="ml_num_of_pc_setup",
                                                type="number",
                                                min=3,
                                                value=6
                                            )
                                        ]),

                                        html.Div(id="ml_random_state_frame", children=[
                                            html.H4(children="Random State"),
                                            dcc.Input(
                                                id="ml_random_state_setup",
                                                type="number",
                                                value=5
                                            )
                                        ]),
                                    ]),

                                    html.Button(id="ml_calc_button_pca_var",
                                                children=[html.Span("Calculate")],
                                                n_clicks=self._pca_calc_trial_num),
                                ]
                            ),

                            html.Div(
                                id="screen201",
                                children=[
                                    html.Div(id="NN_result", children="This is NN Prediction!!!!!"),
                                    html.Div(id="pie_nationality_frame", children=[
                                            dcc.Graph(id="pie_nationality", className="graph"),
                                        ]),
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
        )(self._update_student_prof_department_list)

        self._app.callback(
            dash.dependencies.Output("student_designated_professor", "options"),
            dash.dependencies.Output("student_designated_professor", "value"),
            [dash.dependencies.Input("student_school_list", "value"),
             dash.dependencies.Input("student_department_list", "value")]
        )(self._update_student_prof_designated_professor_name)

        self._app.callback(
            dash.dependencies.Output("student_designated_professor_name", "children"),
            dash.dependencies.Output("student_designated_professor_score", "children"),
            dash.dependencies.Output("student_designated_professor_difficulty", "children"),
            [dash.dependencies.Input("student_school_list", "value"),
             dash.dependencies.Input("student_department_list", "value"),
             dash.dependencies.Input("student_designated_professor", "value")]
        )(self._update_student_designated_professor_overview)

        self._app.callback(
            dash.dependencies.Output("screen11_top_match_score", "children"),
            [dash.dependencies.Input("student_school_list", "value"),
             dash.dependencies.Input("student_department_list", "value"),
             dash.dependencies.Input("student_designated_professor", "value"),
             dash.dependencies.Input("student_top_willing_attributes", "value"),
             dash.dependencies.Input("student_top_unwilling_attributes", "value")]
        )(self._update_screen11_top_match_score)

        self._app.callback(
            dash.dependencies.Output("bar_matched_professors", "figure"),
            [dash.dependencies.Input("student_top_willing_attributes", "value"),
             dash.dependencies.Input("student_top_unwilling_attributes", "value")]
        )(self._update_bar_matched_professors)

        self._app.callback(
            dash.dependencies.Output("prof_department_list", "options"),
            dash.dependencies.Output("prof_department_list", "value"),
            [dash.dependencies.Input("prof_school_list", "value")]
        )(self._update_student_prof_department_list)

        self._app.callback(
            dash.dependencies.Output("prof_designated_professor", "options"),
            dash.dependencies.Output("prof_designated_professor", "value"),
            [dash.dependencies.Input("prof_school_list", "value"),
             dash.dependencies.Input("prof_department_list", "value")]
        )(self._update_student_prof_designated_professor_name)

        self._app.callback(
            dash.dependencies.Output("prof_top_willing_attributes", "options"),
            dash.dependencies.Output("prof_top_willing_attributes", "value"),
            [dash.dependencies.Input("prof_school_list", "value"),
             dash.dependencies.Input("prof_department_list", "value"),
             dash.dependencies.Input("prof_designated_professor", "value")]
        )(self._update_prof_top_willing_attributes)

        self._app.callback(
            dash.dependencies.Output("pca", "figure"),
            dash.dependencies.Output("k-means", "figure"),
            dash.dependencies.Output("ml_calc_button_pca_var", "children"),
            [dash.dependencies.Input("ml_calc_button_pca_var", "n_clicks")],
            [dash.dependencies.State("ml_feature_cols_picker", "value"),
             dash.dependencies.State("ml_num_of_pc_setup", "value"),
             dash.dependencies.State("ml_random_state_setup", "value")]
        )(self._update_pca_clustering_matrix)

        self._app.callback(
            dash.dependencies.Output("pie_nationality", "figure"),
            [dash.dependencies.Input("prof_school_list", "value"),
             dash.dependencies.Input("prof_department_list", "value"),
             dash.dependencies.Input("prof_designated_professor", "value")],
        )(self._update_pie_nationality)

    def _update_student_prof_department_list(self, school_name):
        df = self._data_handler.get_data_frame_original
        department_list = df[df["school_name"] == school_name]["department_name"].unique()
        option_list = [{"label": element, "value": element} for element in department_list]
        value = department_list[0]

        return option_list, value

    def _update_student_prof_designated_professor_name(self, school_name, department_name):
        df = self._data_handler.get_data_frame_original
        professor_list = df[(df["school_name"] == school_name) & (df["department_name"] == department_name)][
            "professor_name"].unique()
        option_list = [{"label": element, "value": element} for element in professor_list]
        value = professor_list[0]

        return option_list, value

    def _update_student_designated_professor_overview(self, school_name, department_name, professor_name):
        df = self._data_handler.get_data_frame_original
        df = df[(df["school_name"] == school_name) & (df["department_name"] == department_name)]
        df = df[["professor_name", "student_star", "student_difficult"]].groupby("professor_name").mean()

        return professor_name, df.loc[professor_name, "student_star"], df.loc[professor_name, "student_difficult"]

    def _update_screen11_top_match_score(self, school_name, department_name, professor_name, willing_list,
                                         unwilling_list):
        professor_id = school_name + department_name + professor_name
        return self._get_matched_score(professor_id, willing_list, unwilling_list)

    def _update_bar_matched_professors(self, willing_list, unwilling_list):
        df = self._data_handler.get_data_frame_original
        df = df[["professor_id", "professor_name"]].drop_duplicates()
        df["matched_score"] = df.apply(
            lambda x: self._get_matched_score(x["professor_id"], willing_list, unwilling_list), axis=1)

        df = df.sort_values(by="matched_score", ascending=False)

        fig = px.bar(data_frame=df, x="professor_name", y="matched_score")
        return fig

    def _update_pca_clustering_matrix(self, trigger, numeric_cols, num_of_pc, random_state):

        # Wait for input fields initialization.
        if not (numeric_cols and num_of_pc and random_state):
            return self._default_plain_fig, self._default_plain_fig

        target_col = "target_grade"
        encoded_cols = self._data_handler.get_all_tag_list
        all_focused_cols = numeric_cols + encoded_cols + [target_col]
        return_list = []

        # Compute PCA
        df_original = self._data_handler.get_data_frame_original[numeric_cols + [target_col, "professor_id"]]
        df_ov = self._data_handler.get_professor_tag_one_hot_vector[encoded_cols].reset_index()
        self._pca_clustering_df["PCA"] = pd.merge(df_original, df_ov, on="professor_id").drop(["professor_id"], axis=1)
        self._pca_clustering_df["PCA"] = self._pca_clustering_df["PCA"].dropna(subset=all_focused_cols).reset_index()

        self._pca_clustering_df["PCA"], total_var = DataHandler.get_pca(data_frame=self._pca_clustering_df["PCA"],
                                                                        numeric_cols=numeric_cols,
                                                                        target=target_col,
                                                                        num_of_pc=num_of_pc)

        # Compute Clustering by using self._pca_clustering_df["PCA"]
        self._pca_clustering_df["K-Means Clustering"] = self._data_handler.get_clustering(
            data_frame=self._pca_clustering_df["PCA"], target=target_col,
            random_state=random_state)

        # Create Matrix figures - Meta
        computed_feature_cols = ["P" + str(i + 1) for i in range(num_of_pc)]

        labels = dict(zip(map(str, range(num_of_pc)), computed_feature_cols))
        labels["color"] = target_col

        # Create Matrix figures
        for key in self._pca_clustering_df.keys():
            return_list.append(
                px.scatter_matrix(
                    self._pca_clustering_df[key],
                    color=self._pca_clustering_df[key][target_col],
                    dimensions=computed_feature_cols,
                    labels=labels,
                    template="simple_white"
                )
            )

            return_list[-1].update_traces(diagonal_visible=False,
                                          marker_coloraxis=None)

            return_list[-1].update_layout(autosize=True,
                                          showlegend=False,
                                          title=key,
                                          margin=go.layout.Margin(l=0, r=0, t=50, b=0),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")

        return_list.append(html.Span(f"Total Explained Variance: {total_var * 100:.2f}%"))

        return tuple(return_list)

    def _update_pie_nationality(self, school_name, department_name, professor_name):
        professor_id = school_name + department_name + professor_name
        nationality_list = ["asian", "hispanic", "nh_black", "nh_white"]
        df = self._data_handler.get_data_frame_original

        df = df[nationality_list + ["professor_id"]]
        df = df[df["professor_id"] == professor_id].groupby(["professor_id"]).mean().reset_index().drop(["professor_id"], axis=1)
        df = df.T.reset_index().rename(columns={"index": "nationality", 0: "percentage"})

        fig = px.pie(df, values="percentage", names="nationality")

        return fig

    @staticmethod
    def _get_color_scale(steps, c_from, c_to):
        return [color.hex for color in list(Color(c_from).range_to(Color(c_to), steps))]

    def _get_matched_score(self, professor_id, willing_list, unwilling_list):
        df = self._data_handler.get_professor_tag_list.loc[[professor_id]]
        professor_score_dict = dict(zip(df.loc[professor_id][0], df.loc[professor_id][1].astype("int32")))

        max_attribute_score = max(len(willing_list), len(unwilling_list))

        willing_score_list = range(max_attribute_score, max_attribute_score - len(willing_list), -1)
        willing_dict = collections.Counter(dict(zip(willing_list, willing_score_list)))

        unwilling_score_list = range(max_attribute_score, max_attribute_score - len(unwilling_list), -1)
        unwilling_dict = collections.Counter(dict(zip(unwilling_list, unwilling_score_list)))

        actual_score = sum([professor_score_dict[key] * willing_dict[key] for key in professor_score_dict.keys()]) - \
                       sum([professor_score_dict[key] * unwilling_dict[key] for key in professor_score_dict.keys()])

        professor_score_list = sorted(professor_score_dict.values(), reverse=True)

        max_score = sum(
            [willing_score_list[i] * (professor_score_list[i] if i < len(professor_score_list) else 1) for i in
             range(len(willing_score_list))])

        return (actual_score / max_score + 1) / 2

    def _update_prof_top_willing_attributes(self, school_name, department_name, professor_name):
        professor_id = school_name + department_name + professor_name
        df = self._data_handler.get_professor_tag_list.loc[[professor_id]]

        value_list = list(df[0])
        option_list = [{"label": element, "value": element} for element in self._data_handler.get_all_tag_list]

        return option_list, value_list

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/RateMyProfessor_Sample data.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
