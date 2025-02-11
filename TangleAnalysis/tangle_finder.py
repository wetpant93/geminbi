from functools import partial
import datasets as ds
import numpy as np
import pandas as pd
from tangles.util.entropy import information_gain
from order_funcs import RatioCutOrder, NormalizedCutOrder, entropy_order
from tangles.convenience import SurveyTangles, Survey, SimpleSurveyFeatureFactory
from tangles.convenience.convenience_orders import create_order_function
from tangles.convenience.convenience_features import compute_corner_features

import sys

sys.setrecursionlimit(10000)


def split_at_true(single_col, invalid_values):
    single_col = np.array(single_col)
    feature = (single_col == True).astype(int) - (single_col == False)
    metadata = np.empty(1, dtype=object)
    metadata[0] = ('==', True)
    return feature[..., np.newaxis], metadata


def create_true_feature_factory(survey):
    feature_factoy = SimpleSurveyFeatureFactory(survey)
    feature_factoy.numvar_func = split_at_true
    return feature_factoy


Dataset = ds.GeminiDatasets.FlashNoComment
data = Dataset.load()
answers = data['answers']
questions = data['questions']

data_frame = pd.DataFrame(answers)
survey = Survey(data_frame)
survey.set_variable_types('numerical')

factory = create_true_feature_factory(survey)
features = factory.create_features()[0]

O1 = create_order_function('O1', features)
O1Ratio = RatioCutOrder(O1)
O1Normal = NormalizedCutOrder(O1)
Entropy = entropy_order
information_gain_order = partial(information_gain, data_frame.to_numpy())
information_gain_ratio = RatioCutOrder(information_gain_order)

AGREEMENT = 50

UNCROSS = False

tangles = SurveyTangles.search(survey,
                               AGREEMENT,
                               feature_factory=create_true_feature_factory(
                                   survey),
                               order=O1Ratio,
                               uncross=UNCROSS)


tangles_dataset = {'tangles': tangles,
                   'questions': questions}

if not UNCROSS:
    save_as = f'{str(Dataset)}-O1Ratio-{AGREEMENT}'
else:
    save_as = f'{str(Dataset)}-O1Ratio-UNCROSS-{AGREEMENT}'


ds.TangleDatasets.save(tangles_dataset, save_as)
