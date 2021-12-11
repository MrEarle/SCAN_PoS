import json
from tqdm import tqdm

pos_map = {
    "<sos>": "<sos>",
    "<eos>": "<eos>",
    "run": "VRB",
    "turn": "VRB",
    "walk": "VRB",
    "jump": "VRB",
    "look": "VRB",
    "right": "DIR",
    "left": "DIR",
    "twice": "ADV",
    "thrice": "ADV",
    "opposite": "MOD",
    "around": "MOD",
    "after": "CNJ",
    "and": "CNJ",
}


# def generate_pos_vocabulary():
#     train, test = load_dataset("simple", split=["train", "test"])
#     pos_vocabulary = set()

#     for sentence in tqdm(train):
#         tags = pos_tag(word_tokenize(sentence["commands"].numpy().decode()), tagset="universal")
#         for _, tag in tags:
#             pos_vocabulary.add(tag)

#     for sentence in tqdm(test):
#         tags = pos_tag(word_tokenize(sentence["commands"].numpy().decode()), tagset="universal")
#         for _, tag in tags:
#             pos_vocabulary.add(tag)

#     with open("src/data/pos_vocab.txt", "w") as f:
#         f.write("\n".join(pos_vocabulary))


def map_pos():
    with open("src/data/pos_ds.json", "r") as f:
        data = json.load(f)

    for key in tqdm(data):
        data[key] = " ".join(pos_map[word] for word in key.split(" "))

    with open("src/data/pos_ds.json", "w") as f:
        json.dump(data, f)

    with open("src/data/pos_vocab.txt", "w") as f:
        f.write("\n".join(set(pos_map.values())))


if __name__ == "__main__":
    with open("src/data/pos_ds.json", "r") as f:
        data = json.load(f)

    max_len = 0
    for key in tqdm(data):
        max_len = max(max_len, len(key.split(" ")))
    print(max_len)
