import numpy as np
from tangles.convenience import SurveyTangles


class tangle_wrapper:
    def __init__(self, tangle):
        self.tangle = tangle
        self.valid_feature_idx = None
        self.tangle_matrix_ = None
        self.ordered_metadata_ = None
        self.__calc_valid()

    def __calc_valid(self):
        tangle_matrix, meta_data = self.tangle.tangle_matrix(
            return_metadata=True)

        meta_data_naked = [str(meta[0].info) for meta in meta_data]
        meta_ordering_naked = [
            str(meta[0].info) for meta in self.tangle.ordered_metadata()]

        ordered_meta_new = np.isin(meta_ordering_naked, meta_data_naked)
        meta_data = np.array(self.tangle.ordered_metadata())[ordered_meta_new]

        self.valid_feature_idx = np.empty(len(meta_data))
        self.ordered_metadata_ = []

        for i, meta in enumerate(meta_data):
            if type(meta[0].info[0]) == str:
                self.valid_feature_idx[i] = True
                self.ordered_metadata_.append(meta)
            else:
                self.valid_feature_idx[i] = False

        self.tangle_matrix_ = tangle_matrix[:,
                                            self.valid_feature_idx.astype(bool)]

    def change_agreement(self, agreement):
        self.tangle.change_agreement(agreement)
        self.__calc_valid()

    def tangle_matrix(self):
        return self.tangle_matrix_

    def ordered_metadata(self):
        return self.ordered_metadata_
