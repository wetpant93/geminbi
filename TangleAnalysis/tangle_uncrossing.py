import pandas as pd
import numpy as np

from functools import partial
from tangles.convenience.survey import Survey
from tangles.convenience.convenience_orders import create_order_function
from tangles.separations.system import SetSeparationSystemOrderFunc, FeatureSystem, MetaData
from tangles.search import uncross_distinguishers, TangleSweep
from tangles.util.graph.similarity import hamming_similarity
from tangles.util.matrix_order import matrix_order
from tangles import agreement_func
from tangles.util.logic import array_to_term
from make_tot import build_tot_sweep, draw_tot, interpret_tot_tangles, grab_and_from_dnf, draw_dnf_tree, draw_dnf_tree_l, get_term_literals
import pickle
from collections import defaultdict

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

    sim_thresh = 0

    agree_mat = -hamming_similarity(feature_matrix,
                                    sim_thresh=sim_thresh).todense().astype(np.int64)

    diag_entries = np.array(-agree_mat.sum(axis=0)).flatten()
    laplacian = np.array(np.diag(diag_entries, k=0) + agree_mat)

    O1 = partial(matrix_order, laplacian)

    # O1 = create_order_function('O1', feat_sys[:])

    feat_sys_ord = SetSeparationSystemOrderFunc(feat_sys, O1)

    sorted_ids = feat_sys_ord.sorted_ids

    sweep = TangleSweep(agreement_func=agreement_func(
        feat_sys), le_func=feat_sys.is_le)

    agreement_value = 50  # 37

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

    print('found efficient distinguishers', efficient_distinguisher_ids)

    for efficient_distinguisher in efficient_distinguisher_ids:
        print('interpreting ', efficient_distinguisher)
        interpret_efficient_distinguisher(
            efficient_distinguisher, feat_sys, feature_matrix, text)

    interpretation_distinguisher = [array_to_term(
        feat_sys[eid], feature_matrix, text) for eid in efficient_distinguisher_ids]

    eff_sweep, eff_feat_sys = build_tot_sweep(feat_sys, efficient_distinguisher_ids,
                                              agreement_value, interpretation_distinguisher)

    meta_expr = interpret_tot_tangles(
        eff_sweep, eff_feat_sys, agreement_value)

    # print(*meta_expr, sep='\n\n\n')

    # for m in meta_expr:
    #     print(*grab_and_from_dnf(m), sep='\n')
    #     print('\n\n')

    lit_dict = dict()
    for i, m in enumerate(meta_expr):
        print(f'Showing tree for tangle {i}')
        # print(m)
        literals = list(map(get_term_literals, grab_and_from_dnf(m)))
        lit_dict[i] = literals
        print(*literals, sep='\n')
        print('\n\n')
        draw_dnf_tree_l(grab_and_from_dnf(m))

    fname = f'lit{agreement_value}_sim{sim_thresh}'
    with open(fname, 'wb') as file:
        pickle.dump(lit_dict, file)

    draw_tot(eff_sweep, agreement_value)


if __name__ == '__main__':
    dataset = GeminiDatasets.Flash15NoCommentSolo

    # data = dataset.load()
    # answers = data['answers']
    # questions = data['questions']

    # data_frame = pd.DataFrame(answers)
    # survey = Survey(data_frame)
    # survey.set_variable_types('numerical')

    # feat_sys = create_feature_system(survey, questions)

    # feature_matrix = feat_sys[:]
    # text = [feat_sys.feature_metadata(i).info.replace(
    #     ' ', '_').replace('.', '').replace('\'', '').replace(',', '') for i in range(len(feat_sys))]

    # agree_mat = -hamming_similarity(feature_matrix,
    #                                 sim_thresh=50).todense().astype(np.int64)

    # diag_entries = np.array(-agree_mat.sum(axis=0)).flatten()
    # laplacian = np.array(np.diag(diag_entries) + agree_mat)

    # O1M = partial(matrix_order, laplacian)

    # O1 = create_order_function('O1', feat_sys[:])

    # feat_sys_ord = SetSeparationSystemOrderFunc(feat_sys, O1)
    # feat_sys_ordM = SetSeparationSystemOrderFunc(feat_sys, O1M)

    tangle_search_uncrossing(dataset=dataset)

    # print(feat_sys_ord.sorted_ids, feat_sys_ordM.sorted_ids, sep='\n\n\n')
