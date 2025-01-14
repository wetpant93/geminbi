import os
import pickle
import time
import google.generativeai as genai


with open('gemini-key.txt', 'r') as file:
    key = file.read()

genai.configure(api_key=key)

name_to_uri = {}
# We divide by 2, since every painting has a descriptor file.
dataset_size = len(os.listdir('../Dataset')) >> 1

for i in range(dataset_size):
    for end in ['.jpg', '.txt']:
        not_uploaded = True
        while not_uploaded:
            try:
                FileUri = genai.upload_file(f'../Dataset/{i}{end}')
                name_to_uri[f'{i}{end}'] = FileUri
                print(f'{i}{end}', ': uploaded')
                not_uploaded = False
            except:
                print(f'I will rest!')
                time.sleep(10)


with open('name_to_uri', 'wb') as file:
    pickle.dump(name_to_uri, file)
