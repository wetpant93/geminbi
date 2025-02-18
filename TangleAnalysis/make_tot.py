import numpy as np
import matplotlib.pyplot as plt
from tangles.util.tree import BinTreeNetworkX
from tangles.separations.system import SetSeparationSystemOrderFunc, FeatureSystem
from tangles.search import TangleSweep
from tangles import agreement_func
from pyeda.inter import expr


def show_tangle_matrix(tangle_matrix: np.ndarray,
                       title: str = None,
                       save_as: str = None):
    """Shows the tangle matrix."""
    if title is not None:
        plt.title(title)

    plt.ylabel("#Of tangles")
    plt.xlabel("#Questions specified")
    plt.imshow(tangle_matrix)
    if save_as is not None:
        plt.savefig(save_as)

    plt.show()


def build_tot_sweep(feat_sys: FeatureSystem, efficient_distinguisher_ids: np.ndarray,
                    agreement: int, efficient_distinguisher_meta: list[str] = None) -> tuple[TangleSweep, FeatureSystem]:

    eff_feat_sys = FeatureSystem(feat_sys.datasize)
    eff_features = feat_sys[efficient_distinguisher_ids]

    if efficient_distinguisher_meta is None:
        eff_features_meta = feat_sys.feature_metadata(
            efficient_distinguisher_ids)

    else:
        eff_features_meta = efficient_distinguisher_meta

    eff_feat_sys.add_features(eff_features, eff_features_meta)

    eff_sweep = TangleSweep(agreement_func=agreement_func(
        eff_feat_sys), le_func=eff_feat_sys.is_le)

    for i in range(len(efficient_distinguisher_ids)):
        print("step", (i+1), "appending", i)
        eff_sweep.append_separation(i, agreement)
        num_max_tangles = len(eff_sweep.tree.k_tangles(
            len(eff_sweep.tree.sep_ids), agreement))
        print("number of leaves left is", num_max_tangles)

    return eff_sweep, eff_feat_sys


def interpret_tot_tangles(sweep: TangleSweep, feat_sys: FeatureSystem, agreement: int):
    matrix = sweep.tree.tangle_matrix(agreement)
    metadata = feat_sys.feature_metadata(sweep.tree.sep_ids)
    meta_expr = []

    for tangle in matrix:
        tangle_meta = []
        for i, ori in enumerate(tangle):
            if ori == 1:
                tangle_meta.append(metadata[i].info.__repr__())
            elif ori == -1:
                tangle_meta.append(f'¬({metadata[i].info.__repr__()})')
            else:
                break
        if tangle_meta is []:
            continue
        tangle_meta = ' and '.join(tangle_meta)
        tangle_meta = tangle_meta.replace(' and ', ' & ').replace(
            ' or ', ' | ').replace('¬', '~')
        meta_expr.append(expr(tangle_meta))

    # print(*meta_expr, sep='\n\n')
    meta_expr = [e.simplify() for e in meta_expr]
    # print('\n\n')
    print(*meta_expr, sep='\n\n')


def draw_tot(sweep: TangleSweep, agreement: int):
    nodes = sweep.tree.maximal_tangles(
        agreement=agreement, include_splitting="nodes")
    tot = BinTreeNetworkX(nodes)
    tot.draw()


if __name__ == "__main__":
    pass
