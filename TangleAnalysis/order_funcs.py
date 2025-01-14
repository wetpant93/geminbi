import numpy as np
from tangles.util.entropy import entropy
from scipy.spatial.distance import pdist, squareform
from tangles.util.matrix_order import matrix_order
from tangles.util.graph.similarity import k_nearest_neighbors
from tangles.util.graph.cut_weight import RatioCutOrder as RCO
from tangles.util.graph.cut_weight_ import NCutOrder as NCO
from tangles.util.graph.similarity import hamming_similarity
from tangles.convenience.convenience_orders import create_order_function


def entropy_order(features):
    """
    Computes the negative discrete entropy order.

    Parameters
    ----------
    Features: np.ndarray -- The features.

    Returns
    -------
    np.ndarray -- The negative entropy of the features.
    """
    return -entropy(features)


def O1_tut(features):
    """
    OrderFunction from the the tangle tutorial.

    Parameters
    ----------
    Features: np.ndarray -- The features.

    Returns
    -------
    np.ndarray -- The order of the features.
    """
    hamming_distance = pdist(features, metric='hamming')
    k_nn_matrix = k_nearest_neighbors(hamming_distance, k=2)
    simularity_matrix = k_nn_matrix + k_nn_matrix.T
    return matrix_order(-simularity_matrix, features)


class RatioCutOrder:
    """
    Given an OrderFunction `f` and a feature {A, B}, this class returns a new OrderFunction `fNew` such that:
                   `fNew`({A, B}) := `f`({A, B}) * (1/|A| + 1/|B|) / 2    

    Parameters
    ----------
    OrderFunction: function -- The OrderFunction to use.
    """

    def __init__(self, order_function):
        self.order_function = order_function

    def __call__(self, features):
        feature_pos = features >= 0
        feature_neg = features <= 0
        factor = (1/feature_pos.sum(axis=0) + 1/feature_neg.sum(axis=0)) * .5
        return self.order_function(features) * factor


class NormalizedCutOrder:
    """
    Given an OrderFunction `f` and a feature {A, B}, this class returns a new OrderFunction `fNew` such that:
                     `fNew`({A, B}) := `f`({A, B}) * (1/A' + 1/B')* 1/2
        where 
            A' := sum( σ(a, b) | (a, b) ∈ (A x V) \ (A x A))
            B' := sum( σ(a, b) | (a, b) ∈ (B x V) \ (B x B))
        Parameters
        ----------
        OrderFunction: function -- The OrderFunction to use
    """

    def __init__(self, order_function):
        self.order_function = order_function

    def __call__(self, features):
        feature_pos = features >= 0
        feature_neg = features <= 0
        M_pos = feature_pos.astype(int) @ feature_pos.transpose()
        M_neg = feature_neg.astype(int) @ feature_neg.transpose()

        summed_simularity = (M_pos.sum(axis=0)
                             + M_neg.sum(axis=0)
                             - feature_pos.sum(axis=1)
                             - feature_neg.sum(axis=1))

        factor = .5 * (1/(summed_simularity @ feature_pos.astype(int))
                       + 1/(summed_simularity @ feature_neg.astype(int)))

        return self.order_function(features) * factor


def index_by_order(order_function, features: np.ndarray) -> np.ndarray:
    """
    Given an OrderFunction and a set of features, this function returns the indices of the features 
    in the order given by the OrderFunction.

    Parameters
    ----------
    OrderFunction: function -- The OrderFunction to use.
    Features: np.ndarray    -- The features.
    """
    ordering = order_function(features)
    indices_by_order = np.argsort(ordering)
    return np.array(indices_by_order)
