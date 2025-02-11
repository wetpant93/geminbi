import datasets as ds
import numpy as np
from tangles.convenience import SurveyTangles
import interesting_tangles as it
import matplotlib.pyplot as plt
import tangle_looker as tl


def shorten_question(question: str) -> str:
    """Removes (True/False) from the end of the question."""
    return question[:-13]


def show_difference(tangle_matrix: np.ndarray,
                    viewer: tl.TangleViewer,
                    title: str = None, save_as: str = None):
    """Shows what questions the tangles in the tangle_matrix have specified differently."""
    if title is not None:
        plt.title(title)

    difference_matrix, tags = viewer.difference_matrix(tangle_matrix)
    plt.imshow(difference_matrix.T)
    plt.yticks(ticks=range(len(tags)),
               labels=[shorten_question(viewer.questions[Tag])
                       for Tag in tags],
               rotation=0)
    plt.yticks(ticks=range(len(difference_matrix.T)))

    if save_as is not None:
        plt.savefig(save_as)

    plt.show()


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


Dataset = ds.TangleDatasets.O1RatioFlashNoComment
data = Dataset.load()
tangles: SurveyTangles = data['tangles']
questions: dict[str, str] = data['questions']
AGREEMENT: int = 75

# alle variabeln in snake_case ; alle klassen in CamelCase
# information gain - order

tangles.change_agreement(AGREEMENT)

viewer = tl.TangleViewer(tangles, questions)
tangle_matrix = tangles.tangle_matrix()

title = f'OrderFunction = {str(Dataset)}, Agreement = {AGREEMENT}'

tangle_table = it.create_tangle_table(tangles, specified=0)
interest_function = it.InterestFunction(tangle_table)
all_cliques = interest_function.get_all_cliques(threshold=2)
max_cliques = interest_function.get_maximal_cliques(all_cliques)
average_specified = dict((clique_size,
                          interest_function.average_specified_questions(max_cliques[clique_size]))
                         for clique_size in max_cliques)

scores = dict((size, interest_function.key_list_order(max_cliques[size]))
              for size in max_cliques)


def show_key_list(key_list):
    """Shows the differences of tangles specified by the key_list."""
    return show_difference(interest_function.key_list_to_tangle_matrix(key_list),
                           viewer,
                           title=title)


show_tangle_matrix(tangle_matrix, title=title)
