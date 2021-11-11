from typing import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

MAX_SEQUENCE_LENGTH = 50


SPLITS = Literal["simple", "addprim_jump", "addprim_turn_left", "length", "mcd1", "mcd2", "mcd3"]

DATA_DIR = "src/data"
IN_VECTORIZER = f"{DATA_DIR}/in_vectorizer"
OUT_VECTORIZER = f"{DATA_DIR}/out_vectorizer"
IN_VOCAB_FILE = f"{DATA_DIR}/in_vocab.txt"
OUT_VOCAB_FILE = f"{DATA_DIR}/out_vocab.txt"

IN_VOCAB_SIZE = 16
OUT_VOCAB_SIZE = 9


def load_dataset(type: SPLITS, data_dir=DATA_DIR, **kwargs):
    return tfds.load(f"scan/{type}", data_dir=data_dir, **kwargs)

def add_sos_eos(text):
    return tf.strings.join(['<SOS>', text, '<EOS>'], separator=' ')

def vectorize(train):
    in_vectorizer = layers.TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int', max_tokens=IN_VOCAB_SIZE)
    out_vectorizer = layers.TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int', max_tokens=OUT_VOCAB_SIZE)

    in_vectorizer.set_vocabulary(IN_VOCAB_FILE)
    out_vectorizer.set_vocabulary(OUT_VOCAB_FILE)

    print('In vocabulary:', in_vectorizer.get_vocabulary())
    print('Out vocabulary:', out_vectorizer.get_vocabulary())

    def vectorize_text(x, y):
        i = in_vectorizer(x)
        o = out_vectorizer(y)

        o = tf.one_hot(o, depth=OUT_VOCAB_SIZE)

        return i, o

    return vectorize_text, in_vectorizer, out_vectorizer

def get_dataset(experiment: SPLITS):
    train, test = load_dataset(experiment, split=['train', 'test'])

    train = train.map(lambda x: (add_sos_eos(x['commands']), add_sos_eos(x['actions'])))
    test = test.map(lambda x: (add_sos_eos(x['commands']), add_sos_eos(x['actions'])))

    vectorize_text, in_vectorizer, out_vectorizer = vectorize(train)

    train = train.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

    train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train, test, (in_vectorizer, out_vectorizer)



if __name__ == "__main__":
    train, test = get_dataset('simple')
    for example in train.take(1):
        print(example)
        break
