import google.generativeai as genai
from gemini_helper import *
from threading import Thread
import time
import pickle


collection = {}
dataset_size = 1001

with open('gemini-key.txt', 'r') as key:
    genai.configure(api_key=key.read())

with open('name_to_uri', 'rb') as file:
    name_to_uri = pickle.load(file)

with open('questioneer_uri', 'rb') as file:
    questioneer_uri = pickle.load(file)


dataset = [[name_to_uri[f'{i}{end}']
            for end in ['.jpg', '.txt']] for i in range(dataset_size)]
dataset_only_jpg = [[name_to_uri[f'{i}.jpg']] for i in range(dataset_size)]


version = 'gemini-1.5-flash-002'


for questioneer_number in range(1, 7):
    collection = {}
    questioneer = questioneer_uri[f'!nq{questioneer_number}.txt']
    threads = [Thread(target=ask_about_datasets, args=([[entry]], questioneer, collection, version))
               for entry in dataset]

    for thread in threads:
        thread.start()
        time.sleep(2)

    for thread in threads:
        thread.join()

    with open(f'./Answerd Questioneers/flash_no_comment_!nq{questioneer_number}', 'wb') as File:
        pickle.dump(collection, File)
