import json
import collections

import pandas as pd
import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc
import plotly.graph_objects as go

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
            mapbox_style=mapbox_info_dict["mapbox_style"]["UCDavis_289H_Final_Project_Spring"]
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
                                    html.Div(id="student_designated_professor_name_frame", children=[
                                        html.Div(id="student_designated_professor_name"),
                                    ]),
                                    html.Div(id="student_designated_professor_info_frame", children=[
                                        html.Div(id="student_designated_professor_score_frame", children=[
                                            html.Div(id="student_designated_professor_score"),
                                            html.Div(id="student_designated_professor_score_description",
                                                     children="Overall Rating")
                                        ]),
                                        html.Div(id="student_designated_professor_difficulty_frame", children=[
                                            html.Div(id="student_designated_professor_difficulty"),
                                            html.Div(id="student_designated_professor_difficulty_description",
                                                     children="Level of Difficulty")
                                        ]),
                                        html.Div(id="screen11_top_match_score_frame", children=[
                                            html.Div(id="screen11_top_match_score"),
                                            html.Div(id="screen11_top_match_score_description",
                                                     children="Professor Matched Percentage")
                                        ]),
                                    ])
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
                                    html.H3(children="Select Professor's School"),
                                    dcc.Dropdown(
                                        id="prof_school_list",
                                        options=[{"label": element, "value": element} for element in
                                                 self._data_handler.get_school_list],
                                        value=self._data_handler.get_school_list[0],
                                        multi=False
                                    ),

                                    html.H3(children="Select Professor's Department"),
                                    dcc.Dropdown(
                                        id="prof_department_list",
                                        multi=False
                                    ),

                                    html.H3(children="Select Designated Professor"),
                                    dcc.Dropdown(
                                        id="prof_designated_professor",
                                        multi=False
                                    ),

                                    html.H3(children="Select Attribute Professor Can Improve"),
                                    dcc.Dropdown(
                                        id="prof_top_willing_attributes",
                                        multi=True
                                    )
                                ]
                            ),

                            html.Div(
                                id="screen201",
                                children=[
                                    html.Div(id="NN_result_frame", children=[
                                        html.Div(id="NN_result_frame_center", children=[
                                            html.Div(id="NN_result_title",
                                                     children="Predicted Score Based on Chosen Attributes"),
                                            html.Div(id="NN_result"),
                                            html.Div(id="NN_result_description", children="Your Score")
                                        ])
                                    ]),

                                    html.Div(id="pie_nationality_frame", children=[
                                        dcc.Graph(
                                            id="pie_nationality",
                                            figure=self._default_plain_fig,
                                            className="graph_style"),
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
                    ),

                    html.Div(
                        id="screen22",
                        children=[
                            html.Div(
                                id="ml_feature_cols_picker_frame",
                                children=[
                                    html.H3(children="PCA/ K-Means Feature Columns Picker"),
                                    dcc.Dropdown(
                                        id="ml_feature_cols_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_numeric_columns],
                                        value=self._data_handler.get_numeric_columns,
                                        multi=True
                                    ),
                                ]
                            ),

                            html.Div(id="ml_num_of_pc_frame", children=[
                                html.H3(children="Principle Columns"),
                                dcc.Input(
                                    id="ml_num_of_pc_setup",
                                    type="number",
                                    min=3,
                                    value=6
                                ),

                                html.H3(children="Total Explained Variance Calculation"),
                                html.Button(id="ml_calc_button_pca_var",
                                            children=[html.Span("Calculate")],
                                            n_clicks=self._pca_calc_trial_num),
                            ]),

                            html.Div(id="ml_random_state_frame", children=[
                                html.H3(children="Random State"),
                                dcc.Input(
                                    id="ml_random_state_setup",
                                    type="number",
                                    value=5
                                ),

                                html.H3(children=self._default_web_credit)
                            ])
                        ]
                    )
                ]
            )
        ])

    def callback(self):
        self._app.callback(
            dash.dependencies.Output("map_rating_heat_map", "figure"),
            [dash.dependencies.Input("screen01_bottom", "children")]
        )(self._update_map_rating_heat_map)

        self._app.callback(
            dash.dependencies.Output("bar_attributes_freq", "figure"),
            [dash.dependencies.Input("map_rating_heat_map", "selectedData")]
        )(self._update_bar_attributes_freq)

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
            dash.dependencies.Output("NN_result", "children"),
            [dash.dependencies.Input("prof_top_willing_attributes", "value")],
        )(self._update_nn_result)

        self._app.callback(
            dash.dependencies.Output("pie_nationality", "figure"),
            [dash.dependencies.Input("prof_school_list", "value"),
             dash.dependencies.Input("prof_department_list", "value"),
             dash.dependencies.Input("prof_designated_professor", "value")],
        )(self._update_pie_nationality)

    def _update_map_rating_heat_map(self, trigger):
        with open("assets/us-states.json") as geojson:
            states = json.load(geojson)

        df = self._data_handler.get_data_frame_original
        df = df[["state_name", "student_star"]].groupby("state_name").mean().reset_index()

        fig = px.choropleth_mapbox(df, geojson=states, locations="state_name", color="student_star",
                                   color_continuous_scale="tealgrn",
                                   range_color=(0, 5),
                                   center={"lat": 37.0902, "lon": -95.7129},
                                   zoom=3,
                                   opacity=0.5
                                   )

        fig.update_layout(
            mapbox=dict(accesstoken=self._mapbox_token, style=self._mapbox_style),
            margin=go.layout.Margin(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        return fig

    def _update_bar_attributes_freq(self, selected_state):
        df_original = self._data_handler.get_data_frame_original
        df_ov = self._data_handler.get_professor_tag_one_hot_vector.drop(
            columns=["tag_professor", "target_grade"]).reset_index()

        if not selected_state or not selected_state["points"]:
            selected_state = list(df_original["state_name"].unique())
        else:
            selected_state = [point["location"] for point in selected_state["points"]]

        state_professor_list = df_original[df_original["state_name"].isin(selected_state)]["professor_id"].unique()
        df_ov = df_ov[df_ov["professor_id"].isin(state_professor_list)].drop(columns=["professor_id"])
        df = df_ov.sum().reset_index().rename(columns={"index": "tags", 0: "frequency"}).sort_values(by="frequency",
                                                                                                     ascending=True)

        fig = px.bar(data_frame=df, x="frequency", y="tags", orientation="h")

        fig.update_traces(
            marker_color="#a5e3dc"
        )

        fig.update_layout(
            margin=go.layout.Margin(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Helvetica",
            font=dict(size=8)
        )

        return fig

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

        rating = df.loc[professor_name, "student_star"]
        difficulty = df.loc[professor_name, "student_difficult"]

        return professor_name, "%.2f" % rating, "%.2f" % difficulty

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

        fig.update_traces(
            marker_color="#a5e3dc"
        )
        fig.update_layout(
            font_family="Helvetica",
            font=dict(size=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=go.layout.Margin(l=10, r=10, t=5)
        )

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
                                          font_family="Helvetica",
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)"
                                          )

        return_list.append(html.Span(f"Total Explained Variance: {total_var * 100:.2f}%"))

        return tuple(return_list)

    def _update_pie_nationality(self, school_name, department_name, professor_name):
        professor_id = school_name + department_name + professor_name
        nationality_list = ["asian", "hispanic", "nh_black", "nh_white"]
        df = self._data_handler.get_data_frame_original

        df = df[nationality_list + ["professor_id"]]
        df = df[df["professor_id"] == professor_id].groupby(["professor_id"]).mean().reset_index().drop(
            ["professor_id"], axis=1)
        df = df.T.reset_index().rename(columns={"index": "nationality", 0: "percentage"})

        fig = px.pie(df, values="percentage", names="nationality", color_discrete_sequence=px.colors.sequential.Purpor,
                     title="Evaluated by (Ethnicity Demographic)")

        fig.update_layout(
            font_family="Helvetica",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    def _update_nn_result(self, selected_attributes):
        X_predict = pd.DataFrame(dict(
            zip(self._data_handler.get_all_tag_list, [[0] for _ in range(len(self._data_handler.get_all_tag_list))])))

        for tag in selected_attributes:
            X_predict.loc[0, tag] = 1

        score = self._data_handler.get_nn_prediction(X_predict)

        return "%.2f" % score

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

        final_score = (actual_score / max_score + 1) * 50

        return "%.1f" % final_score + "%"

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
