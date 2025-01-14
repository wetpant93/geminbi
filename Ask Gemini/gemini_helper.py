from itertools import chain
from threading import Thread
import time
import google.generativeai as genai
from google.generativeai import caching


def process_response(response) -> list[str]:
    """
    Returns the CSV entrys as a list of lowercase strings.

    Parameters
    ----------
    Response -- A CSV Line with separator ','
    """
    answers = response.text.replace('\n', '').lower().split(',')
    return answers


def get_response(entry: genai.File,
                 questioneer: genai.File,
                 model: genai.GenerativeModel,
                 collection: dict[tuple[str, str], list[str]]):
    """Adds a new entry to Collection: 
       Indexed by [Entry.display_name, Questioneer.display_name], 
       contains the answers to the questioneer.

    Args:
        Entry (genai.File): The File that we want the Questioneer answerd for.
        Questioneer (genai.File): Questioneer that is being asked.
        Model (genai.GenerativeModel): The AI that answers the Questioneer.
        Collection (dict[tuple[str,str], list[str]]): A collector for answers.
    """

    entry_name = entry.display_name
    questioneer_name = questioneer.display_name
    prompt = (f'Answer the questions on {questioneer_name} for {entry_name} '
              'in a single line csv format and without any additonal text')

    print(f'requesting for {entry_name} for {questioneer_name}')
    has_response = False
    while not has_response:
        try:
            response = model.generate_content(prompt)
            has_response = True
        except Exception as e:
            print(e)
            print((f'{entry_name} with {questioneer_name} failed,'
                   'trying again after 62 sec'))
            time.sleep(62)

    print(f'{entry_name} for {questioneer_name} is done')
    collection[entry_name, questioneer_name] = process_response(response)


def gather_results(dataset: list[genai.File],
                   questioneer: genai.File,
                   model: genai.GenerativeModel,
                   collection: dict[tuple[str, str], list[str]]):
    """

    Args:
        Dataset (list[genai.File]): A list containting your favorite files!
        Questioneer (genai.File): The Questioneer we want the files to answer.
        Model (genai.GenerativeModel): The AI that answers the Questioneer.
        Collection (dict[tuple[str,str], list[str]]): A collector for answers.
    """

    only_jpg = [filter(lambda File: File.display_name.endswith('.jpg'), entry)
                for entry in dataset]

    only_jpg = chain(*only_jpg)

    threads = [Thread(target=get_response, args=(entry, questioneer, model, collection))
               for entry in only_jpg]

    for thread in threads:
        thread.start()
        time.sleep(0.2)

    for thread in threads:
        thread.join()


def ask_about_datasets(datasets: list[list[genai.File]],
                       questioneer: genai.File,
                       collection: dict[tuple[str, str], list[str]],
                       version: str):
    """Asks the AI model about datasets and collects the responses.

    Args:
        Datasets (list[list[genai.File]]): Your favorite datasets!
        Questioneer (genai.File): The Questioneer you want them to answer.
        Collection (dict[tuple[str,str], list[str]]): A collector for answers.
        Version (str): Version of Gemini that you want to use.
    """

    temperature = genai.GenerationConfig(temperature=0.0)
    for dataset in datasets:
        is_cached = False
        while not is_cached:
            try:
                cache = caching.CachedContent.create(model=version,
                                                     contents=list(chain(*dataset, [questioneer])))
                print('caching complete')

                model = genai.GenerativeModel.from_cached_content(cached_content=cache,
                                                                  generation_config=temperature)
                is_cached = True

            except Exception as e:
                print(f'There was an error {e}\nTrying again after 60 sec')
                time.sleep(60)

        gather_results(dataset, questioneer, model, collection)
        cache.delete()
        print('cache deleted')
