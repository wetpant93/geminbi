import pickle
import google.generativeai as genai 



with open('gemini-key.txt', 'r') as Key:
    genai.configure(api_key=Key.read())

with open('questioneer_uri', 'rb') as File:
    QuestioneerUri = pickle.load(File)
    
for File in QuestioneerUri.values():
    try:
        File.delete()
        print('yippie')
    except:
        print('file is no longer with us')

QuestioneerUri = {}

for i in range(1, 7):
    Path =  f'../Questioneers/!nq{i}.txt'
    File = genai.upload_file(Path)
    FileName = f'!nq{i}.txt'
    QuestioneerUri[FileName] = File
    print(FileName, 'uploaded')

with open('questioneer_uri', 'wb') as File:
    pickle.dump(QuestioneerUri, File)
