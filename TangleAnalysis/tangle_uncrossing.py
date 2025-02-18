import pandas as pd
import numpy as np

from tangles.convenience.survey import Survey
from tangles.convenience.convenience_orders import create_order_function
from tangles.separations.system import SetSeparationSystemOrderFunc, FeatureSystem, MetaData
from tangles.search import uncross_distinguishers, TangleSweep
from tangles import agreement_func
from tangles.util.logic import array_to_term
from make_tot import build_tot_sweep, draw_tot, interpret_tot_tangles


from datasets import GeminiDatasets
from building_features import create_feature_system


def interpret_efficient_distinguisher(feature_id: int, feat_sys: FeatureSystem, feature_matrix: np.ndarray, text: list[str]):
    print(array_to_term(feat_sys[feature_id], feature_matrix, text))


def tangle_search_uncrossing(dataset: GeminiDatasets):
    data = dataset.load()
    answers = data['answers']
    questions = data['questions']

    data_frame = pd.DataFrame(answers)
    survey = Survey(data_frame)
    survey.set_variable_types('numerical')

    feat_sys = create_feature_system(survey, questions)

    feature_matrix = feat_sys[:]
    text = [feat_sys.feature_metadata(i).info.replace(
        ' ', '_').replace('.', '').replace('\'', '').replace(',', '') for i in range(len(feat_sys))]

    O1 = create_order_function('O1', feat_sys[:])

    feat_sys_ord = SetSeparationSystemOrderFunc(feat_sys, O1)

    sorted_ids = feat_sys_ord.sorted_ids

    sweep = TangleSweep(agreement_func=agreement_func(
        feat_sys), le_func=feat_sys.is_le)

    agreement_value = 25

    for i, feature_id in enumerate(sorted_ids):
        print("step", (i+1), "appending", feature_id)
        sweep.append_separation(feature_id, agreement_value)
        print("uncrossing")
        uncross_distinguishers(sweep, feat_sys_ord, agreement_value)
        num_max_tangles = len(sweep.tree.k_tangles(
            len(sweep.tree.sep_ids), agreement_value))
        print("number of leaves left is", num_max_tangles)

    _, efficient_distinguisher_ids = sweep.tree.get_efficient_distinguishers(
        agreement=agreement_value)

    print(feat_sys.feature_metadata(1).type)

    print('found efficient distinguishers', efficient_distinguisher_ids)

    for efficient_distinguisher in efficient_distinguisher_ids:
        print('interpreting ', efficient_distinguisher)
        interpret_efficient_distinguisher(
            efficient_distinguisher, feat_sys, feature_matrix, text)

    interpretation_distinguisher = [array_to_term(
        feat_sys[eid], feature_matrix, text) for eid in efficient_distinguisher_ids]

    eff_sweep, eff_feat_sys = build_tot_sweep(feat_sys, efficient_distinguisher_ids,
                                              agreement_value, interpretation_distinguisher)

    interpret_tot_tangles(eff_sweep, eff_feat_sys, agreement_value)

    draw_tot(eff_sweep, agreement_value)


if __name__ == '__main__':
    dataset = GeminiDatasets.Flash15NoCommentSolo

    tangle_search_uncrossing(dataset=dataset)
