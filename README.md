# Keyphrase Extractor with BERT Embedding Models

## Introduction
One way of understanding and speeding up the consumption of unstructured text by end-users is to automatically extract
so-called “key concepts” and order them in a meaningful way. Performing analytics over such extractions becomes a key 
task to facilitate knowledge acquisition and curation.

## Objective
You will operate on an in-house collection of patent documents. It is your task to parse these patents and to semantically 
enrich them using key concept/phrase extraction so that users can later quickly review documents by looking only at them
instead of reading the text. To support that, you should store the key phrases and document identifiers in a 
data structure or database that supports fast retrieval for analytics purposes.  As an additional use-case, please 
generate a meaningful ordering of the top-30 key phrases also for every document that could help to understand the main theme.

## Dataset 

The XML documents need to be parsed and requires the following steps:
1. [Download](https://databricksexternal.blob.core.windows.net/hiring/patents.zip?sp=r&st=2021-10-07T23:09:03Z&se=2021-10-31T08:09:03Z&spr=https&sv=2020-08-04&sr=b&sig=uR36HP3kCEDY9aPc0mvZFzLnblodA9adxQRTYTc6O6M%3D) the unstructured XML file
2. Unzip (patents file)  in directory Phrase-Extractor-using-KeyBERT/data.

```
cd Phrase-Extractor-using-KeyBERT/src
pip install bs4 absl-py
python parser.py
```
Note: Download and parsing should be done before building the docker image (~1 hour depends on Sys config)

### Environment variables & installations

1. First, clone repository and then run the following commands
```
cd Phrase-Extractor-using-KeyBERT
docker build -f Dockerfile -t docker_key_extractor .
```

2. Once the docker image is built successfully and python library installations are successful.
```
docker run -ti docker_key_extractor
```

3. Activate the virtual environment
```
source /venv/bin/activate
```

4. If parsing is already done or Phrase-Extractor-using-KeyBERT/data/raw is available, run the following
```
cd KPE/src
python3 keyBERT.py 
```
optionally as provided in flags description in keyBERT.py
```
python3 keyBERT.py --model_name [model name] 
```


## Solution

Keyword Extraction: process of extracting most relevant words/phrases from an input text. 
Exisiting approaches like [YAKE](#https://github.com/LIAAD/yake) and [Rake](#https://github.com/aneesha/RAKE) work on 
statistical approaches which fail to capture the semantic structure in natural languages. 
BERT - a bi-directional transformer model converts text into embedding vectors such that they can capture the context of 
a document. A detailed tutorial on how BERT embeddings are used for keyword extraction models can be found [here](#https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea)


In this solution the core idea can be split into the following:

1. Candidate Keywords/Keyphrases --> controlled by n_gram_range
2. BERT Embeddings --> good performance for both similarity- and paraphrasing tasks. Available options [SpaCy](#https://spacy.io/), [Hugginface transformers](#https://github.com/huggingface/transformers),
[Flair](#https://github.com/flairNLP/), [sentence-transformers](#https://www.sbert.net/).
3. Cosine Similarity --> compare the document and candidate embeddings
4. Diversification --> if keywords needs to be diversified (two options available Max Sum Similarity/Maximal Margin Relevance)