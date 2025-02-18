import tangle_looker as tl
import numpy as np
from tangles.util.logic import array_to_term
from tangles.convenience import SurveyTangles


def p(tangles: SurveyTangles, agreement: int):
    feat_sys = tangles.feature_system
    tangle_sweep = tangles.sweep._sweep

    all_mat, all_nodes = tangle_sweep.tree.tangle_matrix(
        agreement=agreement, include_splitting="nodes", return_nodes=True)

    max_mat, max_nodes = tangle_sweep.tree.tangle_matrix(
        agreement=agreement, include_splitting="nope", return_nodes=True)

    idx = np.count_nonzero(max_mat, axis=0) != np.sum(np.abs(max_mat), axis=0)

    feature_ids = tangle_sweep.tree.sep_ids[:, idx]


def interpret_tags(tangles: SurveyTangles, questions: dict[str, str]) -> SurveyTangles:
    feat_sys = tangles.feature_system
    ids = tangles.sweep.original_feature_ids
    text = [questions[meta.info[0]]
            for meta in feat_sys.feature_metadata(ids)]

    for feat_id in feat_sys.all_sep_ids():
        new_meta = array_to_term(feat_sys[feat_id], feat_sys[:], text)
