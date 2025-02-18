from functools import partial
import datasets as ds
import numpy as np
import pandas as pd
import tangles
from tangles.util.entropy import information_gain
from order_funcs import RatioCutOrder, NormalizedCutOrder, entropy_order
from tangles.convenience import SurveyTangles, Survey, SimpleSurveyFeatureFactory
from tangles.convenience.convenience_orders import create_order_function
from tangles.convenience.convenience_features import compute_corner_features
from tangles.separations import FeatureSystem
from tangle_wrapper import tangle_wrapper
from tangles.search import TangleSweep
from tangles.util.tree import BinTreeNode
from tangles.search.progress import DefaultProgressCallback

from building_features import create_feature_meta_pair

import sys

sys.setrecursionlimit(10000)

if __name__ == '__main__':
    Dataset = ds.GeminiDatasets.Flash15NoCommentSolo
    data = Dataset.load()
    answers = data['answers']
    questions = data['questions']

    data_frame = pd.DataFrame(answers)
    survey = Survey(data_frame)
    survey.set_variable_types('numerical')

    features, feature_meta = create_feature_meta_pair(survey)

    O1 = create_order_function('O1', features)
    O1Ratio = RatioCutOrder(O1)
    O1Normal = NormalizedCutOrder(O1)
    Entropy = entropy_order
    information_gain_order = partial(information_gain, data_frame.to_numpy())
    information_gain_ratio = RatioCutOrder(information_gain_order)

    order_names = {
        O1: 'O1',
        O1Ratio: 'O1Ratio',
        O1Normal: 'O1Normal',
        Entropy: 'Entropy',
        information_gain_order: 'InformationGain',
        information_gain_ratio: 'InformationGainRatio'
    }

    selected_order = information_gain_order
    order_name = order_names[selected_order]

    feature_orders = selected_order(features)
    sorted_feature_idx = np.argsort(feature_orders)

    corners, corner_meta = compute_corner_features(
        features, order_func=selected_order,
        min_side_size=100, max_order_factor=2, global_max_order=feature_orders[sorted_feature_idx][(len(feature_orders) - 1)//3])

    all_features = np.append(features, corners, axis=1)
    all_meta = np.append(feature_meta, corner_meta)

    original_feature_ids = list(range(features.shape[1]))
    corner_feature_ids = list(
        range(len(original_feature_ids) + corners.shape[1]))

    sep_sys = FeatureSystem.with_array(
        all_features, metadata=all_meta)

    for meta in sep_sys.feature_metadata(corner_feature_ids):
        meta.type = 'inf'

    AGREEMENT = 50
    UNCROSS = False

    tangles = SurveyTangles.search(survey,
                                   agreement=AGREEMENT,
                                   features_or_separations=sep_sys,
                                   order=selected_order,
                                   uncross=UNCROSS)

    tangles.sweep.original_feature_ids = original_feature_ids

    tangles_dataset = {'tangles': tangles,
                       'questions': questions}

    if not UNCROSS:
        save_as = f'{str(Dataset)}-{order_name}-{AGREEMENT}'
    else:
        save_as = f'{str(Dataset)}-{order_name}-UNCROSS-{AGREEMENT}'

    ds.TangleDatasets.save(tangles_dataset, save_as)
