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


Dataset = ds.TangleDatasets.O1
data = Dataset.load()
tangles: SurveyTangles = data['tangles']
questions: dict[str, str] = data['questions']
AGREEMENT: int = 50

# alle variabeln in snake_case ; alle klassen in CamelCase
# information gain - order

tangles.change_agreement(AGREEMENT)

Viewer = tl.TangleViewer(tangles, questions)
TangleMatrix = tangles.tangle_matrix()

Title = f'OrderFunction = {str(Dataset)}, Agreement = {AGREEMENT}'

tangle_table = it.create_tangle_table(tangles, specified=50)
InterestFunction = it.InterestFunction(tangle_table)
all_cliques = InterestFunction.get_all_cliques(threshold=50)
max_cliques = InterestFunction.get_maximal_cliques(all_cliques)
average_specified = dict((Size, InterestFunction.average_specified_questions(max_cliques[Size]))
                         for Size in max_cliques)

scores = dict((size, InterestFunction.key_list_order(max_cliques[size]))
              for size in max_cliques)


def show_key_list(key_list):
    return show_difference(InterestFunction.key_list_to_tangle_matrix(key_list),
                           Viewer,
                           title=Title)


show_tangle_matrix(TangleMatrix, title=Title)

show_key_list(max_cliques[3])
