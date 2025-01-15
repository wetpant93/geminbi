from collections import defaultdict
from itertools import combinations
import numpy as np
import networkx as nx
from tangles.convenience import SurveyTangles


def get_tangle_range(Range: tuple[int, int], tangles: SurveyTangles) -> np.ndarray:
    """
    Let Range = (`x1`, `x2`):
        Returns all tangles that have specified atleast `x1` and atmost `x2` many questions.

    Parameters
    ----------
    Range: tuple[int, int]
    Tangles: SurveyTangles

    """
    lower_bound, upper_bound = Range
    tangle_matrix = tangles.tangle_matrix()

    count_specified = np.abs(tangle_matrix).sum(axis=1)

    valid_indices = (lower_bound <= count_specified).astype(
        int) & (count_specified <= upper_bound).astype(int)

    return tangle_matrix[valid_indices.astype(bool)]


def create_tangle_table(tangles: SurveyTangles, specified: int | tuple[int, int] = 0) -> dict[int, np.ndarray]:
    """
    If Specfied is an int, say `x1`:
        Returns a dict of all tangles that have specified atleast `x1` many questions.

    If Specfied is a pair of ints, say (`x1`, `x2`):
        Returns a dict of all tangles that have specified atleast `x1` and atmost `x2` many questions.

    The keys of the returned dict, starting from 0, are the enumerated tangles that fullfilled the condition
    imposed by Specfied.

    Parameters
    ----------
    Tangles: SurveyTangles
    Specified: int | tuple[int, int]
    """
    if isinstance(specified, int):
        tangle_matrix = tangles.tangle_matrix()
        tangle_matrix = tangle_matrix[specified <=
                                      (tangle_matrix != 0).sum(axis=1)]
    else:
        tangle_matrix = get_tangle_range(specified, tangles)

    return dict(enumerate(tangle_matrix))


class InterestFunction:
    def __init__(self, tangle_table: dict[int, np.ndarray]):
        """
        Creates an InterestFunction object, that calculates the amount of diff

        Parameters
        ----------
        TangleTable: dict[int, np.ndarray] -
        """
        if tangle_table is []:
            raise ValueError("TangleTable can\'t be empty")

        # Checks if all elements of TangleTable have the same size
        tangles = list(tangle_table.values())

        if any(len(tangles[0]) != len(tangle) for tangle in tangles):
            raise ValueError(
                "Not all elements of TangleTable have the same size")

        self.tangle_length: int = len(tangles[0])
        self.tangle_table: dict[int, np.ndarray] = tangle_table

        self.intrest_score_table: dict[int, int] = dict()
        self.interest_score_to_pairs: dict[int,
                                           list[(int, int)]] = defaultdict(list)

        for (first_key, second_key) in combinations(tangle_table, 2):
            self(first_key, second_key)

    def __call__(self, first_key: int, second_key: int) -> int:
        """
        Given FirstKey and SecondKey of the specified TangleTable, computes the amount of questions that the tangles:
            t1 = TangleTable[FirstKey] and t2 = TangleTable[SecondKey]
        have specified differently.
        Questions that atleast one of the tangles did not specify are not counted.

        Parameters
        ----------
        FirstKey: int   -- Key of TangleTable
        SecondKey: int  -- Key of TangleTable
        """
        if interest_score := self.intrest_score_table.get((first_key, second_key)) is not None:
            return interest_score

        first_tangle = self.tangle_table.get(first_key)
        second_tangle = self.tangle_table.get(second_key)

        if first_tangle is None:
            raise ValueError(f'The FirstKey: {first_key}'
                             + 'is not a valid key of TangleTable')

        elif second_tangle is None:
            raise ValueError(f'The SecondKey: {second_key}'
                             + 'is not a valid key of TangleTable')

        count_zeros_first_tangle = np.count_nonzero(first_tangle == 0)
        count_zeros_second_tangle = np.count_nonzero(second_tangle == 0)
        maximal_zeros_counted = max(
            count_zeros_first_tangle, count_zeros_second_tangle)

        if maximal_zeros_counted == 0:
            interest_score = self.tangle_length - \
                (first_tangle == second_tangle).sum()
        else:
            interest_score = (self.tangle_length
                              - maximal_zeros_counted
                              - (first_tangle[:-maximal_zeros_counted] == second_tangle[:-maximal_zeros_counted]).sum())

        self.intrest_score_table[first_key, second_key] = interest_score
        self.intrest_score_table[second_key, first_key] = interest_score
        self.interest_score_to_pairs[interest_score].extend(
            [(first_key, second_key), (second_key, first_key)])

        return interest_score

    def key_list_to_tangle_matrix(self, key_list: list[int]) -> np.ndarray:
        """
        Returns a TangleMatrix consisting of tangles specified by keys of KeyList of TangleTable.

        Parameters
        ----------
        KeyList: list[int] -- A list of keys of the specified TangleTable.
        """
        return np.array([self.tangle_table[Key] for Key in key_list])

    def get_pairs_threshold(self, threshold: int) -> list[(int, int)]:
        """
        Returns a list of all pairs of keys (FirstKey, SecondKey) such that: 
            InterestFunction(FirstKey, SecondKey) ≥ Threshold

        Parameters
        ----------
        Threshold: int -- Number of questions two Tangles should specfiy differently.
        """
        return [pair for interest_score in self.interest_score_to_pairs
                for pair in self.interest_score_to_pairs[interest_score]
                if interest_score >= threshold]

    def key_list_order(self, key_list: list[int]) -> float:
        """
        Computes the average amount of questions specified differently between 
        tangles of TangleTable specified by KeyList

        Parameters
        ----------
        KeyList: list[int] -- A list of keys of the specified TangleTable.
        """
        if len(key_list) <= 1:
            return 0

        elif any(Key not in self.tangle_table.keys() for Key in key_list):
            raise ValueError('KeyList has a key not in TangleTable')

        length = len(key_list)

        interest_score_all_pairs = sum(self.intrest_score_table[FirstKey, SecondKey]
                                       for (FirstKey, SecondKey) in combinations(key_list, 2))

        number_of_pairs = length * (length - 1) * .5

        return interest_score_all_pairs / number_of_pairs

    def average_specified_questions(self, key_list: list[int]) -> float:
        """
        Returns the average amount of specified questions, of tangles in TangleTable specfied by KeyList.

        Parameters
        ----------
        KeyList: list[int] -- A list of keys of the specified TangleTable.
        """

        if key_list is []:
            return 0

        return sum([np.count_nonzero(self.tangle_table[Key]) for Key in key_list])/len(key_list)

    def get_all_cliques(self, threshold: int = 0, key_list: list[int] = None) -> dict[int, list[int]]:
        """Returns all cliques of the following graph `G`:
            Vertices of `G` are the tangles in TangleTable specfied by KeyList; 
                If KeyList = None, then: 
                    V(`G`) = [TangleTable[Key] for Key in TangleTable.keys()]
                Else:                  
                    V(`G`) = [TangleTable[Key] for Key in KeyList].

            Two vertices of `G` say `t1`, `t2` ∈ V(`G`) share an edge, if and only if:
                `t1` and `t2` specfiy atleast Threshold many questions differently.

        Parameters
        ----------
            Threshold (int): Amount of questions two tangles should specfiy differently.
            KeyList (list[int], optional): List of keys specifying which tangles to include in the graph. 
                                           Defaults to None.

        Returns
        -------
            dict[int, list[int]]: A dictionary where the keys are the clique sizes found, and the values are lists of cliques of that size. 
                                  Each clique is represented as a list of keys from the TangleTable.
        """
        graph = nx.Graph()

        if key_list is None:
            graph.add_nodes_from(self.tangle_table.keys())
        else:
            graph.add_nodes_from(key_list)

        graph.add_edges_from(self.get_pairs_threshold(threshold))
        all_cliques = defaultdict(list)

        for clique in nx.find_cliques(graph):
            all_cliques[len(clique)].append(clique)

        return all_cliques

    def get_maximal_cliques(self, cliques: dict[int, list[int]], order=None) -> dict[int, list[int]]:
        """
        Gives back a dictionary of maximal cliques of Cliques with respect to Order.
        If Order is None, then InterestFunction.KeyListOrder is used.

        Parameters
        ----------
        Cliques : dict[int, list[int]] -- 
            A dictionary where the keys are the clique sizes found, 
            and the values are lists of cliques of that size. 
            Each clique is represented as a list of keys from the TangleTable.

        Order : function, optional -- A function that takes a list of keys and returns a number. 
                                      Defaults to None.

        Returns
        -------
        dict[int, list[int]]: 
            A dictionary where the keys are the clique sizes found, 
            and the values are the keys of the maximal cliques of that size.

        """
        if order is None:
            key = self.key_list_order
        else:
            key = order

        maximal_cliques = dict(
            (Size, max(cliques[Size], key=key)) for Size in cliques)

        return maximal_cliques
