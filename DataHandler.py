import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


class DataHandler(object):

    def __init__(self, data_frame):
        self._data_frame_original = data_frame
        self._data_frame_last_update = None

        self._attribute_list = {"gives_good_feedback", "caring", "respected", "participation_matters", "clear_grading_criteria", "skip_class", "amazing_lectures", "inspirational", "tough_grader", "hilarious", "get_ready_to_read", "lots_of_homework", "accessible_outside_class", "lecture_heavy", "extra_credit", "graded_by_few_things", "group_projects", "test_heavy", "so_many_papers", "beware_of_pop_quizzes", "IsCourseOnline"}
        self._school_list = self._data_frame_original["school_name"].unique()
        self._professor_list = self._data_frame_original["professor_name"].unique()

        self.preprocess_data()

    @classmethod
    def construct_from_csv(cls, path):
        return DataHandler(data_frame=pd.read_csv(path, encoding="ISO-8859-1"))

    def preprocess_data(self):
        clean_list = [
            "tag_professor"
        ]

        # clean
        self._data_frame_last_update = self._data_frame_original.replace(to_replace="", value=np.nan)
        self._data_frame_last_update = self._data_frame_last_update.dropna(subset=clean_list).reset_index()

        self._data_frame_last_update["professor_id"] = self._data_frame_last_update["school_name"] + self._data_frame_last_update["department_name"] + self._data_frame_last_update["professor_name"]

        # Apply changes to original data_frame
        self.apply_last_update()

    @staticmethod
    def get_pca(data_frame, target, num_of_pc):
        x = StandardScaler().fit_transform(data_frame.loc[:, data_frame.columns != target])

        pca = PCA(n_components=num_of_pc)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=["P" + str(i + 1) for i in range(num_of_pc)])
        df_pca = pd.concat([principal_df, data_frame[target]], axis=1)

        return df_pca, pca.explained_variance_ratio_.sum()

    @staticmethod
    def get_clustering(data_frame, target, random_state):
        x = scale(data_frame.loc[:, data_frame.columns != target])
        y = data_frame[target]

        clustering = KMeans(n_clusters=len(y.unique()), random_state=random_state)
        clustering.fit(x)

        df_clustering = data_frame.copy()
        df_clustering[target] = clustering.labels_
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
    def get_attribute_list(self):
        return self._attribute_list

    @property
    def get_school_list(self):
        return self._school_list

    @property
    def get_department_list(self):
        return self._department_list

    @property
    def get_professor_list(self):
        return self._professor_list


def unit_test():
    path = "dataset/RateMyProfessor_Sample data.csv"
    handler = DataHandler.construct_from_csv(path=path)
    print(handler.get_data_frame_original)


if __name__ == "__main__":
    unit_test()
