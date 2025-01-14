from enum import StrEnum
import pickle
from os.path import isfile
from tangles.convenience import SurveyTangles


class TangleDatasets(StrEnum):
    O1 = 'TangleDatasets/Flash-O1-50'
    O1Ratio = 'TangleDatasets/Flash-O1Ratio-50'
    Entropy = 'TangleDatasets/Flash-Entropy-50'

    def __str__(self):
        """
        Returns the name of the dataset. For example, TangleDatasets.O1 will return 'O1'.
        If the dataset is not recognized, it will return None.
        """
        match self:
            case TangleDatasets.O1:
                return "O1"
            case TangleDatasets.O1Ratio:
                return "O1Ratio"
            case TangleDatasets.Entropy:
                return "Entropy"
            case _:
                return None

    @staticmethod
    def save(tangles_dataset: dict[str, SurveyTangles | dict[str, str]], save_as: str):
        """
        Save a TanglesDataset to a file.

        Parameters
        ----------
        TanglesDataset : dict[str, SurveyTangles | dict[str, str]]
            The dataset to save.
        SaveAs : str
            The name of the file that will hold the dataset.
        """

        if isfile(f'TangleDatasets/{save_as}'):
            raise ValueError('Such a file already exists')

        with open(f'TangleDatasets/{save_as}', 'wb') as File:
            pickle.dump(tangles_dataset, File)

    def load(self) -> dict[str, SurveyTangles | dict[str, str]]:
        """
        Loads a TanglesDataset. Gives back a dictionary with the keys 'tangles' and 'questions'.
        where:
            'tangles' is a SurveyTangles object
            'questions' is a dictionary with the questions indexed by tags.

        Parameters
        ----------
        self : TangleDatasets
            The dataset to load.
        """
        if self in [TangleDatasets.O1,
                    TangleDatasets.O1Ratio,
                    TangleDatasets.Entropy]:
            TanglePath = self.value
            with open(TanglePath, 'rb') as File:
                Dataset = pickle.load(File)

            return Dataset

        else:
            raise ValueError('There is no such dataset.')


class GeminiDatasets(StrEnum):
    Flash = 'GeminiDatasets/flash_!nq'
    Pro = 'GeminiDatasets/pro_!nq_pro'
    FlashNoComment = 'GeminiDatasets/flash_no_commment_!nq'
    ProNoComment = 'GeminiDatasets/pro_no_comment_!nq'

    def __str__(self):
        """
        Returns the name of the dataset. For example, GeminiDatasets.Flash will return 'Flash'.
        If the dataset is not recognized, it will return None.
        """
        match self:
            case GeminiDatasets.Flash:
                return "Flash"
            case GeminiDatasets.Pro:
                return "Pro"
            case GeminiDatasets.FlashNoComment:
                return "FlashNoComment"
            case GeminiDatasets.ProNoComment:
                return "ProNoComment"
            case _:
                return None

    def load(self) -> dict[str, dict[str, list[int]] | dict[str, str]]:
        """
        Loads a GeminiDataset. Gives back a dictionary with the keys 'answers' and 'questions'.
        where:
            'answers' is a dictionary with the answers indexed by tags.
            'questions' is a dictionary with the questions indexed by tags.

        Parameters
        ----------
        self : GeminiDatasets
            The dataset to load.
        """
        if self in [GeminiDatasets.Flash,
                    GeminiDatasets.Pro,
                    GeminiDatasets.FlashNoComment,
                    GeminiDatasets.ProNoComment]:
            dataset_path = self.value
            with open(dataset_path, 'rb') as file:
                return pickle.load(file)

        raise ValueError('There is no such dataset')
