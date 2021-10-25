import os
import re
import time
import pickle
import torch
from absl import app
from absl import flags
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

FLAGS = flags.FLAGS
flags.DEFINE_enum('diversification', 'mmr', ['mmr', 'mms'], "Candidate keyphrase diversification metric")  # set to False if extraction not required
flags.DEFINE_float('diversity', 0.5, 'Diversity value (only for mmr)')
flags.DEFINE_integer('n_grams', 2, 'Length of candidate key phrases')
flags.DEFINE_integer('top_n', 30, 'Top N key-phrases')
flags.DEFINE_integer('nr_candidates', 30, 'Total candidates to be considered')
flags.DEFINE_enum('model_name', 'all-MiniLM-L12-v2', ['all-MiniLM-L12-v2', 'all-MiniLM-L6-v2', 'distilbert-base-nli-mean-tokens',
                                                      'all-distilroberta-v1', 'all-mpnet-base-v2'], 'Sentence Transformer Embedding Model')

def regex(text):
    text = re.sub('[\n\d\']', '', text)  # remove new lines and numerical characters
    text = re.sub('\W+', ' ', text)  # replace non-alphabets with empty space
    text = re.sub('\s+', ' ', text)  # remove extra whitespaces
    return text.lower()


def preprocessing(root_dir):
    '''
    Text preprocessing using regular expressions
    :param root_dir: base directory for the project
    :return: None; dumps the cleaned text in processed_dir
    '''

    raw_dir = os.path.join(root_dir, 'raw')
    # create a processed directory
    processed_dir = os.path.join(root_dir, 'processed')
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    # change current working directory
    os.chdir(raw_dir)
    directories = [f for f in list(os.listdir()) if '.json' in f]  # considers only unzipped directories

    for dir in directories:  # iterate on each raw JSON object
        print('_' * 100)
        print(f'Pre-processing on {dir} ...')
        # load each raw JSON object
        with open(os.path.join(raw_dir, dir), 'rb') as fp:
            my_dict = pickle.load(fp)

        # perform text pre-processing on each XML doc identifier
        keys, values = list(my_dict.keys()), list(my_dict.values())
        values = list(map(regex, values))

        # save the processed text again as dictionary object
        my_dict = dict(zip(keys, values))
        with open(os.path.join(processed_dir, dir), 'wb') as fp:
            pickle.dump(my_dict, fp)

def key_phraser(doc: list, model) -> dict:
    '''
    Extracts key phrases for the batch data and model passed
    :param doc: batch of text documents
    :param model: Sentence Transformer model
    :return: batch of key-phrases/concepts
    '''

    n_grams = (1, 2)
    nr_candidates = 30
    top_n = 30
    diversity = 0.5
    diversification = 'mmr'

    if diversification == 'mss':  # max-sum-similarity
        candidates = model.extract_keywords(doc, keyphrase_ngram_range=n_grams, stop_words='english',
                                            use_maxsum=True, nr_candidates=nr_candidates, top_n=top_n)
    elif diversification == 'mmr':  # maximum-marginal-relevance
        candidates = model.extract_keywords(doc, keyphrase_ngram_range=n_grams, stop_words='english',
                                            use_mmr=True, nr_candidates=nr_candidates, top_n=top_n, diversity=diversity)
    return candidates

def main(argv):

    root_dir = os.path.join(os.getcwd(), '../data')
    preprocessing(root_dir)
    # make a new 'key_phrases' directory if does not exists
    key_phrase_dir = os.path.join(root_dir, 'key_phrases')
    if not os.path.isdir(key_phrase_dir):
        os.makedirs(key_phrase_dir)

    processed_dir = os.path.join(root_dir, 'processed')
    os.chdir(processed_dir)
    jsonfiles = [f for f in list(os.listdir()) if '.json' in f]  # considers all processed files
    jsonfiles = ['ongoing_100001_120000.json', 'ongoing_140001_160000.json']
    for jsonfile in jsonfiles:  # read each JSON binary file
        print('_' * 100)
        print(f'Generating key phrases for {jsonfile} ...')
        start = time.time()
        with open(os.path.join(processed_dir, jsonfile), 'rb') as fp:
            my_dict = pickle.load(fp)

        doc_ids, docs = list(my_dict.keys()), list(my_dict.values())
        docs_batch = torch.utils.data.DataLoader(dataset=docs, batch_size=2048,
                                                 num_workers=multiprocessing.cpu_count())

        try:
            candidates = []
            # create a model instance
            model_name = FLAGS.model_name
            sentence_model = SentenceTransformer(model_name)
            model = KeyBERT(model=sentence_model)  # based on tradeoff b/w metrics (in https://www.sbert.net/docs/pretrained_models.html) and speed
            # forward (batch processing)
            for batch_id, doc in enumerate(docs_batch):
                candidates.append(key_phraser(doc, model))
                print(f'Batch: {batch_id + 1}')

            print('Time:', time.time() - start)
            # dictionary data structure saves memory
            candidates_flatten = [elem for candidate in candidates for elem in candidate]
            candidates_flatten = [dict(sorted({elem[1]: elem[0] for elem in candidate}.items(), reverse=True)) for candidate in candidates_flatten]
            my_dict = [dict(zip(cf.values(), cf.keys())) for cf in candidates_flatten]  # reverse the keys and values

            # save key phrases in a new directory
            with open(os.path.join(key_phrase_dir, jsonfile), 'wb') as fp:  # pickle as binary object
                pickle.dump(my_dict, fp)
            print(f'Key phrases saved for {jsonfile} successfully')
        except ValueError as ve:  # a single file doe not have any textual content upon parsing
            print(ve)

if __name__ == '__main__':
    app.run(main)