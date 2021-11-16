import json
from nltk.tag import pos_tag, pos_tag_sents
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from .scan import load_dataset


def generate_pos_vocabulary():
    train, test = load_dataset('simple', split=['train', 'test'])
    pos_vocabulary = set()

    for sentence in tqdm(train):
        tags = pos_tag(word_tokenize(sentence['commands'].numpy().decode()), tagset='universal')
        for _, tag in tags:
            pos_vocabulary.add(tag)

    for sentence in tqdm(test):
        tags = pos_tag(word_tokenize(sentence['commands'].numpy().decode()), tagset='universal')
        for _, tag in tags:
            pos_vocabulary.add(tag)

    with open('src/data/pos_vocabulary.txt', 'w') as f:
        f.write('\n'.join(pos_vocabulary))

def map_pos(path, output_path):
    with open(path, 'r') as f:
        exs = [word_tokenize(line.strip()) for line in f]

    pos = pos_tag_sents(exs, tagset='universal')

    res = {}
    for pos_list in pos:
        ex, tags = zip(*pos_list)
        res[" ".join(ex)] = " ".join(tags)
    
    with open(output_path, 'w') as f:
        json.dump(res, f)