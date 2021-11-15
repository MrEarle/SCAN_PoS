import json
from typing import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

from ..utils.constants import *
from ..utils.args import args

MAX_SEQUENCE_LENGTH = 50


SPLITS = Literal["simple", "addprim_jump", "addprim_turn_left", "length", "mcd1", "mcd2", "mcd3"]

DATA_DIR = "src/data"

IN_VECTORIZER = f"{DATA_DIR}/in_vectorizer"
OUT_VECTORIZER = f"{DATA_DIR}/out_vectorizer"

IN_VOCAB_FILE = f"{DATA_DIR}/in_vocab.txt"
POS_VOCAB_FILE = f"{DATA_DIR}/pos_vocab.txt"
OUT_VOCAB_FILE = f"{DATA_DIR}/out_vocab.txt"

DS_POS_FILE = f"{DATA_DIR}/pos_ds.json"

IN_VOCAB_SIZE = 15
POS_VOCAB_SIZE = 9
OUT_VOCAB_SIZE = 8


def load_dataset(type: SPLITS, data_dir=DATA_DIR, **kwargs):
    return tfds.load(f"scan/{type}", data_dir=data_dir, **kwargs)

def get_add_pos_tag(ds_path: str):
    with open(ds_path, 'r') as f:
        pos_ds = json.load(f)

    def add_pos_tag(in_text, out_text):
        pos_tag = pos_ds[in_text.numpy().decode()]
        return in_text, tf.convert_to_tensor(pos_tag), out_text

    return add_pos_tag

def add_sos_eos(in_text, pos_text, out_text):
    in_text =  tf.strings.join(['<sos>', in_text, '<eos>'], separator=' ')
    pos_text =  tf.strings.join(['<sos>', pos_text, '<eos>'], separator=' ')
    out_text =  tf.strings.join(['<sos>', out_text, '<eos>'], separator=' ')
    return in_text, pos_text, out_text

def get_vectorizer():
    in_vectorizer = layers.TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int', max_tokens=IN_VOCAB_SIZE, standardize=None)
    pos_vectorizer = layers.TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int', max_tokens=POS_VOCAB_SIZE, standardize=None)
    out_vectorizer = layers.TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int', max_tokens=OUT_VOCAB_SIZE, standardize=None)

    in_vectorizer.set_vocabulary(IN_VOCAB_FILE)
    pos_vectorizer.set_vocabulary(POS_VOCAB_FILE)
    out_vectorizer.set_vocabulary(OUT_VOCAB_FILE)

    print('In vocabulary:', { i: x for i,x in enumerate(in_vectorizer.get_vocabulary())})
    print('Pos vocabulary:', { i: x for i,x in enumerate(pos_vectorizer.get_vocabulary())})
    print('Out vocabulary:', { i: x for i,x in enumerate(out_vectorizer.get_vocabulary())})

    def vectorize_text(x, p, y):
        x = in_vectorizer(x)
        p = pos_vectorizer(p)
        y = out_vectorizer(y)

        p = tf.one_hot(p, depth=POS_VOCAB_SIZE)
        y = tf.one_hot(y, depth=OUT_VOCAB_SIZE)

        return x, p, y

    return vectorize_text, in_vectorizer, pos_vectorizer, out_vectorizer

def get_dataset(experiment: SPLITS):
    train, test = load_dataset(experiment, split=['train', 'test'])

    # Split dataset into tuples
    train = train.map(lambda x: (x['commands'], x['actions']))
    test = test.map(lambda x: (x['commands'], x['actions']))

    add_pos_tag = get_add_pos_tag(DS_POS_FILE)

    # Add pos tag to dataset
    train = train.map(lambda x, y: tf.py_function(add_pos_tag, inp=(x, y), Tout=(tf.string, tf.string, tf.string)))
    test = test.map(lambda x, y: tf.py_function(add_pos_tag, inp=(x, y), Tout=(tf.string, tf.string, tf.string)))

    # Add <sos> and <eos> to dataset
    train = train.map(add_sos_eos)
    test = test.map(add_sos_eos)

    vectorize_text, in_vectorizer, pos_vectorizer, out_vectorizer = get_vectorizer()

    # Vectorize dataset
    train = train.map(lambda x, p, y: tf.py_function(vectorize_text, inp=(x,p,y), Tout=(tf.int64, tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(lambda x, p, y: tf.py_function(vectorize_text, inp=(x,p,y), Tout=(tf.int64, tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)

    if args.include_pos_tag == 'aux':
        # If pos tag is used in auxiliary function
        train = train.map(lambda x, p, y: ({COMMAND_INPUT_NAME: x}, {ACTION_OUTPUT_NAME: y, POS_OUTPUT_NAME: p}))
        test = test.map(lambda x, p, y: ({COMMAND_INPUT_NAME: x}, {ACTION_OUTPUT_NAME: y, POS_OUTPUT_NAME: p}))
    elif args.include_pos_tag == 'input':
        # If pos tag is used as input
        train = train.map(lambda x, p, y: ({COMMAND_INPUT_NAME: x, POS_INPUT_NAME: p}, {ACTION_OUTPUT_NAME: y}))
        test = test.map(lambda x, p, y: ({COMMAND_INPUT_NAME: x, POS_INPUT_NAME: p}, {ACTION_OUTPUT_NAME: y}))
    else:
        # No pos tag
        train = train.map(lambda x, _, y: ({COMMAND_INPUT_NAME: x}, {ACTION_OUTPUT_NAME: y}))
        test = test.map(lambda x, _, y: ({COMMAND_INPUT_NAME: x}, {ACTION_OUTPUT_NAME: y}))

    train = train.cache().prefetch(buffer_size=1)
    test = test.cache().prefetch(buffer_size=1)

    return train, test, (in_vectorizer, pos_vectorizer, out_vectorizer)



if __name__ == "__main__":
    train, test = get_dataset('simple')
    for example in train.take(1):
        print(example)
        break
