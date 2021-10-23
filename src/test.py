

pth = '/Users/sharma/PycharmProjects/Key-Phrase-Extractor/data/processed/ongoing_1_20000.json'

with open(pth, 'rb') as fp:
    my_dict = pickle.load(fp)

keys, values = list(my_dict.keys()), list(my_dict.values())
doc = values[0]


x =






print(candidates)