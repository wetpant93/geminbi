import pickle
from collections import defaultdict
from itertools import dropwhile


def string_to_int(string: str) -> int:
    return 1 if string.lower() == 'true' else 0


def to_key(entry_index: int, questioneer_index: int) -> tuple[str, str]:
    return f'{entry_index}.jpg', f'!nq{questioneer_index}.txt'


DATASET_SIZE = 1001  # Size of the Dataset
DATASET = 'flash_!nq'  # Name of the Dataset

# Define tags for questions in the questioneers, each question in them is numberd.
# Example:
# QuestionTags = {
#    1 : [(1, 'S'), (2, 'S')],
#    2 : [],
#    ...   ,
# }
# Gives the questions 1 and 2 in the 1st questioneer the the tag 'S'.

question_tags: dict[int, list[(int, str)]] = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}

# Default tag for a Question, that has not defined a tag.
DEFAULT_TAG = 'Q'
freq_map = defaultdict(int)    # Counts how often a tag is used.

# Will hold the question corrosponding to a tag.
questions_by_tags: dict[str, str] = {}
# Will hold the answers from Gemini, indexed by the tags of questions.
answerd_questions: dict[str, list[int]] = {}
merged_dataset = {'answers': answerd_questions,
                  'questions': questions_by_tags}


# Questions that you want to exclude from a questioneer.
# Example:
# UnusedQuestions = {
#    1 : [1, 2, 3, 5, 8],
#    2 : [],
#    ...   ,
# }
# removes the questions 1,2,3,5,8 from the 1st questioneer from the merged datasets.

unused_questions: dict[int, list[int]] = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}


for questioneer_index in unused_questions:
    with open(f'{DATASET}{questioneer_index}', 'rb') as File:
        answers = pickle.load(File)

    with open(f'../Questioneers/!nq{questioneer_index}.txt', 'r') as File:
        questioneer = File.read().split('\n')
        questioneer = [''.join(dropwhile(lambda char: not char.isalpha(), question))
                       for question in questioneer
                       if question != '']

    index_to_tags = dict(question_tags[questioneer_index])
    for question_index in range(len(questioneer)):
        if tag := index_to_tags.get(question_index) is None:
            tag = DEFAULT_TAG

        freq_map[tag] += 1

        key = tag + str(freq_map[tag])

        answerd_questions[key] = [string_to_int(answers[to_key(entry_index, questioneer_index)][question_index])
                                  for entry_index in range(DATASET_SIZE)]

        questions_by_tags[key] = questioneer[question_index]


with open(f'MergedData/{DATASET}', 'wb') as File:
    pickle.dump(merged_dataset, File)
