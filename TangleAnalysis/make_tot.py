import numpy as np
import matplotlib.pyplot as plt
from tangles.util.tree import BinTreeNetworkX
from tangles.separations.system import SetSeparationSystemOrderFunc, FeatureSystem
from tangles.search import TangleSweep
from tangles import agreement_func
from tangles.util.logic import TextTerm
from functools import reduce
import operator
from pyeda.inter import expr
from itertools import combinations
from collections import defaultdict
import re
import networkx as nx


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
        eff_feat_sys), le_func=eff_feat_sys.is_le, forbidden_tuple_size=2)

    for i in range(len(efficient_distinguisher_ids)):
        print("step", (i+1), "appending", i)
        eff_sweep.append_separation(i, agreement)
        num_max_tangles = len(eff_sweep.tree.k_tangles(
            len(eff_sweep.tree.sep_ids), agreement))
        print("number of leaves left is", num_max_tangles)

    return eff_sweep, eff_feat_sys


def textterm_to_expr(text: TextTerm):
    text_str = text.__repr__()
    clean_text = text_str.replace(' and ', ' & ').replace(
        ' or ', ' | ').replace('¬', '~')
    return expr(clean_text)


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

        def to_expression_string(meta): return meta.replace(' and ', ' & ').replace(
            ' or ', ' | ').replace('¬', '~')

        to_expressions = map(lambda meta: expr(
            to_expression_string(meta)).simplify(), tangle_meta)

        def join_and_simplify(expr1, expr2): return operator.and_(
            expr1, expr2).to_dnf().simplify()

        reduced_tangle_expr = reduce(join_and_simplify, to_expressions)
        # tangle_meta = ' and '.join(tangle_meta)
        # tangle_meta = tangle_meta.replace(' and ', ' & ').replace(
        #     ' or ', ' | ').replace('¬', '~')
        meta_expr.append(reduced_tangle_expr)

    # print(*meta_expr, sep='\n\n')
    # print('\n\n')
    # print(*meta_expr, sep='\n\n')

    return meta_expr


def grab_and_from_dnf(expression):
    pattern = "And\([^\(\)]+\)"
    return re.findall(pattern, str(expression))


def get_term_literals(expression):
    pattern = "[^\(\),]+"
    res = re.findall(pattern, str(expression.replace(' ', '')))
    if len(res) == 1:
        return res

    return res[1:]


def draw_dnf_tree(dnf_terms):
    root = "root"
    label_dict = {root: 'root'}
    tree = nx.DiGraph()
    tree.add_node(root)
    where = dict()
    dnf_terms = list(map(lambda term: set(
        get_term_literals(term)), dnf_terms))

    # print(*dnf_terms, sep='\n\n')
    # print()
    # print()
    frq = defaultdict(int)
    where[root] = dnf_terms
    while any(where.values()):
        current_keys = where.copy().keys()
        for key in current_keys:
            terms = where[key]
            # print('terms:')
            # print(*terms, sep='\n\n')
            i = len(terms)
            for i in range(len(terms), 0, -1):
                for combination in combinations(terms, i):
                    # print('combination:')
                    # print(*combination, sep='\n\n')
                    res = list(reduce(operator.__and__, combination))
                    # print('combination:')
                    # print(*combination, sep='\n\n')
                    if len(res) > 0:
                        break
                else:
                    continue
                #   print(*combination, sep='\n\n')

                # print('combination:')
                # print(*combination, sep='\n\n')
                for i, node in enumerate(res):
                    if frq[node] != 0:
                        res[i] = node + f'{frq[node]}'
                    frq[node] += 1
                    break_when = len(res[i]) // 2
                    label_dict[res[i]] = res[i][:break_when] + \
                        '\n' + res[i][break_when:]

                tree.add_nodes_from(res)
                tree.add_edges_from((v, w)
                                    for v, w in zip([key] + res, res))
                sres = set(res)
                new_comb = []
                for comb in combination:
                    if len(combination) != 1:
                        new_comb.append(comb - sres)
                        # print(comb, "COMB")
                        # print(sres, "SRES")
                    where[key].remove(comb)
                where[res[-1]] = new_comb
                break

        # print(where.values())

    # for term in dnf_terms:
    #     current_node = root
    #     for lit in term:
    #         if lit not in tree.nodes():
    #             tree.add_node(lit)
    #         tree.add_edge(current_node, lit)
    #         current_node = lit

    pos = nx.drawing.nx_agraph.graphviz_layout(
        tree, prog='dot')

    fig, ax = plt.subplots(figsize=(30, 30))
    nx.draw(tree, pos, ax, labels=label_dict, node_size=700, node_color="skyblue",
            font_size=7, arrowsize=20, arrowstyle='-|>')

    plt.show()


def get_successor(node, current_node, tree):
    for succ in tree.successors(current_node):
        if node in succ:
            return succ
    return None


def draw_dnf_tree_l(dnf_terms):
    dnf_terms = list(map(get_term_literals, dnf_terms))
    tree = nx.DiGraph()
    root = 'root'
    labels_dict = {root: 'root'}

    node_frq = defaultdict(int)
    tree.add_node(root)
    for term in dnf_terms:
        current_node = root

        while len(term) > 0:
            for node in term:
                if successor := get_successor(node, current_node, tree):
                    break

            term.remove(node)
            if successor is None:
                frq = node_frq[node]
                new_node = node + f'{frq}'
                break_point = len(node) // 2
                labels_dict[new_node] = node[:break_point] + \
                    '\n' + node[break_point:]

                tree.add_node(new_node)
                tree.add_edge(current_node, new_node)
                node_frq[node] += 1
                current_node = new_node
            else:
                current_node = successor

    pos = nx.drawing.nx_agraph.graphviz_layout(
        tree, prog='dot', args='-Gscale=2')

    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(tree, pos, ax, labels=labels_dict, node_size=700, node_color="skyblue",
            font_size=10, arrowsize=20, arrowstyle='-|>')

    plt.show()


def and_string_to_terms(string):
    pattern = "[^\(\),]"
    return re.findall(pattern, string)


def draw_tot(sweep: TangleSweep, agreement: int):
    nodes = sweep.tree.maximal_tangles(
        agreement=agreement, include_splitting="nodes")
    tot = BinTreeNetworkX(nodes)
    tot.draw()


if __name__ == "__main__":
    pass
