import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


class DataHandler(object):

    def __init__(self, data_frame):
        self._data_frame_original = data_frame
        self._data_frame_last_update = None

        self._professor_tag_list = None
        self._all_tag_list = None
        self._professor_tag_one_hot_vector = None
        self._school_list = self._data_frame_original["school_name"].unique()

        self._numeric_cols = ["star_rating", "diff_index", "num_student", "student_star", "student_difficult", "word_comment", "asian", "hispanic", "nh_black", "nh_white"]

        self._clustering_model = None

        self.preprocess_data()

    @classmethod
    def construct_from_csv(cls, path):
        return DataHandler(data_frame=pd.read_csv(path, encoding="ISO-8859-1"))

    def preprocess_data(self):
        mlb = MultiLabelBinarizer()
        clean_list = [
            "tag_professor"
        ]

        # Clean
        self._data_frame_original = self._data_frame_original.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self._data_frame_last_update = self._data_frame_original.replace(to_replace=["", "unknown", "NAN"], value=np.nan)
        self._data_frame_last_update = self._data_frame_last_update.dropna(subset=clean_list).reset_index()

        # Create professor_id
        self._data_frame_last_update["professor_id"] = self._data_frame_last_update["school_name"] + \
                                                       self._data_frame_last_update["department_name"] + \
                                                       self._data_frame_last_update["professor_name"]

        # Calculate professor tag list & all tags
        df = self._data_frame_last_update.copy()
        df = df[["professor_id", "tag_professor", "student_star"]].groupby(["professor_id", "tag_professor"]).mean().reset_index().set_index("professor_id")
        df = df.rename(columns={"student_star": "target_grade"})

        self._professor_tag_list = df["tag_professor"].str.extractall(r"(.*?)\s*\((.*?)\)\s*")
        self._all_tag_list = list(self._professor_tag_list[0].unique())

        # Calculate one-hot-vector vs student_star
        dff = self._professor_tag_list.groupby(by=["professor_id"], axis=0)[0].apply(list).reset_index(name="tag_list").set_index("professor_id")
        self._professor_tag_one_hot_vector = pd.concat([df, pd.DataFrame(mlb.fit_transform(dff["tag_list"]), columns=mlb.classes_, index=dff.index)], axis=1)

        df = df.round({"target_grade": 0}).astype({"target_grade": "int32"}).astype({"target_grade": "str"})

        self._data_frame_last_update = pd.merge(self._data_frame_last_update, df, on="professor_id")

        # Apply changes to original data_frame
        self.apply_last_update()

    @staticmethod
    def get_pca(data_frame, numeric_cols, target, num_of_pc):
        x = data_frame.copy()
        x[numeric_cols] = StandardScaler().fit_transform(data_frame[numeric_cols])

        pca = PCA(n_components=num_of_pc)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=["P" + str(i + 1) for i in range(num_of_pc)])
        df_pca = pd.concat([principal_df, data_frame[target]], axis=1)

        return df_pca, pca.explained_variance_ratio_.sum()

    def get_clustering(self, data_frame, target, random_state):
        x = scale(data_frame.loc[:, data_frame.columns != target])
        y = data_frame[target]

        self._clustering_model = KMeans(n_clusters=len(y.unique()), random_state=random_state)
        self._clustering_model.fit(x)

        df_clustering = data_frame.copy()
        df_clustering[target] = self._clustering_model.labels_
        return df_clustering

    def apply_last_update(self):
        self._data_frame_original = self._data_frame_last_update.copy()

    def reset_last_update(self):
        self._data_frame_last_update = self._data_frame_original.copy()

    @staticmethod
    def get_top_categories(data_frame, target_cols, number_of_reserved=8):
        return list(data_frame[target_cols].value_counts().head(number_of_reserved).index)

    @staticmethod
    def trim_categories(data_frame, target_cols, designated_list=None, number_of_reserved=None):
        df = data_frame.copy()

        reserved_categories = designated_list if designated_list else data_frame[target_cols].value_counts().index
        df.loc[~data_frame[target_cols].isin(reserved_categories[:number_of_reserved]), target_cols] = "Other"

        return df

    @property
    def get_data_frame_original(self):
        return self._data_frame_original

    @property
    def get_data_frame_last_update(self):
        return self._data_frame_last_update

    @property
    def get_professor_tag_list(self):
        return self._professor_tag_list

    @property
    def get_all_tag_list(self):
        return self._all_tag_list

    @property
    def get_professor_tag_one_hot_vector(self):
        return self._professor_tag_one_hot_vector

    @property
    def get_school_list(self):
        return self._school_list

    @property
    def get_numeric_columns(self):
        return self._numeric_cols


def unit_test():
    path = "dataset/RateMyProfessor_Sample data.csv"
    handler = DataHandler.construct_from_csv(path=path)
    print(handler.get_all_tag_list)


if __name__ == "__main__":
    unit_test()
