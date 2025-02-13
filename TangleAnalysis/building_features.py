import numpy as np

from tangles.convenience.survey_feature_factory import SimpleSurveyFeatureFactory
from tangles.convenience.survey import Survey
from tangles.separations import FeatureSystem

def _split_at_true(single_col, _):
    single_col = np.array(single_col)
    feature = (single_col == True).astype(int) - (single_col == False)
    metadata = np.empty(1, dtype=object)
    metadata[0] = ('==', True)
    return feature[..., np.newaxis], metadata


def _create_true_feature_factory(survey):
    feature_factoy = SimpleSurveyFeatureFactory(survey)
    feature_factoy.numvar_func = _split_at_true
    return feature_factoy

def create_feature_system(survey: Survey, questions: dict[str, str]) -> FeatureSystem:
    factory = _create_true_feature_factory(survey)
    features, feature_meta = factory.create_features()
    for i, meta in enumerate(feature_meta):
        a, b, c = meta
        assert b == '==', f'something weird happened with {meta}'
        assert c == True, f'something weird happened with {meta}'
        assert a in questions, f'weeeeird key {a} not in dictionary'
        feature_meta[i] = questions[a].replace(" (True/False)", '')
    return FeatureSystem.with_array(features, metadata=feature_meta)