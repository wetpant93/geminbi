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


Dataset = ds.GeminiDatasets.Flash15NoComment
data = Dataset.load()
answers = data['answers']
questions = data['questions']

data_frame = pd.DataFrame(answers)
survey = Survey(data_frame)
survey.set_variable_types('numerical')

factory = create_true_feature_factory(survey)
features, feature_meta = factory.create_features()


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

features = features.T[sorted_feature_idx].T
feature_meta = feature_meta[sorted_feature_idx]

corners, corner_meta = compute_corner_features(
    features, order_func=selected_order,
    min_side_size=250, max_order_factor=2, global_max_order=feature_orders[sorted_feature_idx][(len(feature_orders) - 1)//2])

# print(selected_order(corners), feature_orders, sep='\n\n')


all_features = np.append(corners, features, axis=1)
all_meta = np.append(corner_meta, feature_meta)
all_features_by_size = np.argsort(selected_order(all_features))


# all_features = features
# all_meta = feature_meta
# all_features_by_size = np.argsort(selected_order(all_features))

# corner_index_by_size = np.argsort(selected_order(corners))

all_features_sorted = all_features.T[all_features_by_size].T
all_meta_sorted = all_meta[all_features_by_size]


sep_sys = FeatureSystem.with_array(
    all_features_sorted, metadata=all_meta_sorted)


AGREEMENT = 50
UNCROSS = False


# a_func = tangles.agreement_func(sep_sys=sep_sys)

# tsweep = tangles.TangleSweep(a_func, le_func=sep_sys.is_le,
#                              sep_ids=sep_sys.all_sep_ids(), forbidden_tuple_size=3)


# tsweep.sweep_below(agreement=AGREEMENT,
#                    progress_callback=DefaultProgressCallback())


tangles = SurveyTangles.search(survey,
                               agreement=AGREEMENT,
                               features_or_separations=sep_sys,
                               order=selected_order,
                               uncross=UNCROSS)


tangles = tangle_wrapper(tangles)


tangles_dataset = {'tangles': tangles,
                   'questions': questions}

if not UNCROSS:
    save_as = f'{str(Dataset)}-{order_name}-{AGREEMENT}'
else:
    save_as = f'{str(Dataset)}-{order_name}-UNCROSS-{AGREEMENT}'


ds.TangleDatasets.save(tangles_dataset, save_as)
