import numpy as np
from tangles.convenience import SurveyTangles


class TangleViewer:
    def __init__(self, tangles: SurveyTangles, questions: list[str]):

        self.tangles = tangles
        self.questions = questions
        _, meta = self.tangles.tangle_matrix(return_metadata=True)
        self.max_tags = len(meta)
        self.tag_order = np.array([m[0].info[0] for m in meta])

    def show_tangle(self, tangle: np.ndarray,
                    surpress_non_answer: bool = True,
                    only_show_tag: bool = True):
        """Prints how Tangle has specified Questions.

            Parameters
            ----------
            Tangle (np.ndarray): An array which tells us how the Questions are specified.
            SurpressNonAnswer (bool, optional): If True, skips over non specified Questions. 
                                                Defaults to True.
            OnlyShowTag (bool, optional): If True, doesn't print the full question, only it's tag. 
                                          Defaults to True.
        """

        for tag, answer in zip(self.tag_order, tangle):
            if surpress_non_answer and not answer:
                break
            if only_show_tag:
                print(tag, answer)
            else:
                print(self.questions[tag], answer)

    def show_difference(self, first_tangle: np.ndarray,
                        second_tangle: np.ndarray):
        """Prints out the questions that the Tangles have specified differently.
           Questions that are printed out are specified by both Tangles.

            Parameters
            ----------
            FirstTangle (np.ndarray): An array which tells us how the Questions are specified.
            SecondTangle (np.ndarray): An array which tells us how the Questions are specified.
        """

        count_non_zero_first_tangle = np.count_nonzero(first_tangle)
        count_non_zero_second_tangle = np.count_nonzero(second_tangle)

        minimum_non_zero = min(count_non_zero_first_tangle,
                               count_non_zero_second_tangle)
        select = first_tangle != second_tangle
        select[minimum_non_zero:] = 0

        for select_able, tag in list(zip(select, self.tag_order))[:minimum_non_zero]:
            if select_able:
                print(self.questions[tag])

    def show_questions_by_order(self):
        """
        Prints questions to screen by their order.
        """
        for order, tag in enumerate(self.tag_order):
            print(order, tag, self.questions[tag])

    def tag_to_order(self, tag: str) -> int:
        """
        Parameters
        ----------
        Tag (str): Tag of a question.

        Returns
        -------
        int: The absolute order of a tag.
        """
        if tag not in self.tag_order:
            raise ValueError("There is no such Tag.")

        for order, current_tag in enumerate(self.tag_order):
            if current_tag == tag:
                return order

    def difference_matrix(self, tangle_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray[str]]:
        """Gives back the columns of the TangleMatrix 
           that have at least two differing non-zero entries, 
           along with their corresponding tags.

        Parameters
        ----------
        TangleMatrix (np.ndarray): 
            A 2D array where each row represents a tangle and each column represents a question.

        Returns
        -------
        tuple[np.ndarray, np.ndarray[str]]: A tuple containing two elements:
            - A 2D array with the selected columns from the original TangleMatrix.
            - A 1D array of tags corresponding to the selected columns.
        """
        index = np.count_nonzero(tangle_matrix, axis=0) != np.abs(
            np.sum(tangle_matrix, axis=0))
        tags = self.tag_order[index]
        result = (tangle_matrix.T[index]).T
        return result, tags
