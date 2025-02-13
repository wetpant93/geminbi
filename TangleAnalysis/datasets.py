from enum import StrEnum
import pickle
from os.path import isfile
from tangles.convenience import SurveyTangles
from tangle_wrapper import tangle_wrapper


class TangleDatasets(StrEnum):
    O1Flash = 'TangleDatasets/Flash-O1-50'
    O1RatioFlash = 'TangleDatasets/Flash-O1Ratio-50'
    EntropyFlash = 'TangleDatasets/Flash-Entropy-50'
    InformationGainFlash = 'TangleDatasets/Flash-InformationGain-50'
    O1FlashNoComment = 'TangleDatasets/FlashNoComment-O1-50'
    O1ProNoCommentUncross = 'TangleDatasets/ProNoComment-O1-UNCROSS-15'
    O1ProNoComment = 'TangleDatasets/ProNoComment-O1-50'
    InformationGainRatioFlash = 'TangleDatasets/Flash-InformationGainRatio-50'
    O1Flash15NoComment = 'TangleDatasets/Flash15NoComment-O1-60'
    O1RatioFlashNoComment = 'TangleDatasets/FlashNoComment-O1Ratio-50'
    O1RatioFlash15NoComment = 'TangleDatasets/Flash15NoComment-O1Ratio-50'
    InformationGainFlash15NoComment = 'TangleDatasets/Flash15NoComment-InformationGain-50'
    InformationGainFlash15NoCommentUncross = 'TangleDatasets/Flash15NoComment-InformationGain-UNCROSS-30'

    def __str__(self):
        """
        Returns the name of the dataset. For example, TangleDatasets.O1 will return 'O1'.
        If the dataset is not recognized, it will return None.
        """
        match self:
            case TangleDatasets.O1Flash:
                return "O1Flash"
            case TangleDatasets.O1RatioFlash:
                return "O1RatioFlash"
            case TangleDatasets.EntropyFlash:
                return "EntropyFlash"
            case TangleDatasets.InformationGainFlash:
                return "InformationGainFlash"
            case TangleDatasets.O1FlashNoComment:
                return "O1FlashNoComment"
            case TangleDatasets.O1ProNoCommentUncross:
                return "O1ProNoCommentUncross"
            case TangleDatasets.O1ProNoComment:
                return "O1ProNoComment"
            case TangleDatasets.InformationGainRatioFlash:
                return "InformationGainRatioFlash"
            case TangleDatasets.O1RatioFlashNoComment:
                return "O1RatioFlashNoComment"
            case TangleDatasets.O1RatioFlash15NoComment:
                return "O1RatioFlash15NoComment"
            case TangleDatasets.InformationGainFlash15NoComment:
                return "InformationGainFlash15NoComment"
            case TangleDatasets.InformationGainFlash15NoCommentUncross:
                return "InformationGainFlash15NoCommentUncross"
            case TangleDatasets.O1Flash15NoComment:
                return "O1Flash15NoComment"
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
        if self in [TangleDatasets.O1Flash,
                    TangleDatasets.O1RatioFlash,
                    TangleDatasets.EntropyFlash,
                    TangleDatasets.InformationGainFlash,
                    TangleDatasets.O1FlashNoComment,
                    TangleDatasets.O1ProNoCommentUncross,
                    TangleDatasets.O1ProNoComment,
                    TangleDatasets.InformationGainRatioFlash,
                    TangleDatasets.O1RatioFlashNoComment,
                    TangleDatasets.O1RatioFlash15NoComment,
                    TangleDatasets.InformationGainFlash15NoComment,
                    TangleDatasets.InformationGainFlash15NoCommentUncross,
                    TangleDatasets.O1Flash15NoComment]:

            tangle_path = self.value
            with open(tangle_path, 'rb') as file:
                dataset = pickle.load(file)

            return dataset

        else:
            raise ValueError('There is no such dataset.')


class GeminiDatasets(StrEnum):
    Flash = 'GeminiDatasets/flash_!nq'
    Pro = 'GeminiDatasets/pro_!nq_pro'
    FlashNoComment = 'GeminiDatasets/flash_no_comment_!nq'
    ProNoComment = 'GeminiDatasets/pro_no_comment_!nq'
    Flash15NoComment = 'GeminiDatasets/flash15_no_comment_!nq'

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
            case GeminiDatasets.Flash15NoComment:
                return "Flash15NoComment"
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
                    GeminiDatasets.ProNoComment,
                    GeminiDatasets.Flash15NoComment]:
            dataset_path = self.value
            with open(dataset_path, 'rb') as file:
                return pickle.load(file)

        raise ValueError('There is no such dataset')
