import pickle
from collections import defaultdict
from itertools import dropwhile, chain
from operator import add


def to_key(i, j):
    return f'{i}.jpg', f'!nq{j}.txt'


def str_answerd(i, j, questioneer, collection):
    answers = collection[to_key(i, j)]
    return '\n'.join(map(lambda x: ' '.join(x), zip(questioneer, answers)))


def merge_answers(answers):
    merged = defaultdict(list)
    for name, index in answers.keys():
        merged[name].append((index, answers[name, index]))

    for name in merged.keys():
        sorted(merged[name])
        merged[name] = list(chain(*[answer for _, answer in merged[name]]))

    return merged


def string_to_int(string):
    if string.lower() == 'true':
        return 1
    return 0


def sum_col(merged_answers):
    result = [0] * len(list(merged_answers.values())[0])
    for name in merged_answers.keys():
        bit_list = list(map(string_to_int, merged_answers[name]))
        result = list(map(lambda Pair: add(*Pair), zip(bit_list, result)))

    return list(result)


def check_if_all_same_size(answerd_questions):
    length_of_answers = list(map(len, answerd_questions.values()))
    return all(length_of_answers[0] == Entry for Entry in length_of_answers)


N = 3
DATASET = f'flash_!nq{N}'


questions = []
answerd_questions = []

with open(f'../Questioneers/!nq{N}.txt', 'r') as file:
    Questioneer = file.read().split('\n')
    Questioneer = [''.join(dropwhile(lambda Char: not Char.isalpha(), Question))
                   for Question in questions
                   if Question != '']
    questions.extend(Questioneer)


with open(DATASET, 'rb') as file:
    answerd_questions = pickle.load(file)

if check_if_all_same_size(answerd_questions):
    print('All answers are the same size!')

I = answerd_questions
answerd_questions = merge_answers(answerd_questions)


print(*zip(range(1, 100), sum_col(answerd_questions)), sep='\n')
