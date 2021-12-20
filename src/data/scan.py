import json
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

from ..utils.constants import *

MAX_SEQUENCE_LENGTH = 50
MAX_IN_SEQUENCE_LENGTH = 12
MAX_OUT_SEQUENCE_LENGTH = 51


SPLITS = Literal["simple", "addprim_jump", "addprim_turn_left", "length", "mcd1", "mcd2", "mcd3"]

DATA_DIR = "src/data"

IN_VECTORIZER = f"{DATA_DIR}/in_vectorizer"
OUT_VECTORIZER = f"{DATA_DIR}/out_vectorizer"

IN_VOCAB_FILE = f"{DATA_DIR}/in_vocab.txt"
POS_VOCAB_FILE = f"{DATA_DIR}/pos_vocab.txt"
OUT_VOCAB_FILE = f"{DATA_DIR}/out_vocab.txt"

DS_POS_FILE = f"{DATA_DIR}/pos_ds.json"

IN_VOCAB_SIZE = 15
POS_VOCAB_SIZE = 7
OUT_VOCAB_SIZE = 8


def load_dataset(type: SPLITS, data_dir=DATA_DIR, **kwargs):
    return tfds.load(f"scan/{type}", data_dir=data_dir, **kwargs)


def get_add_pos_tag(ds_path: str):
    with open(ds_path, "r") as f:
        pos_ds = json.load(f)

    def add_pos_tag(in_text, out_text):
        pos_tag = pos_ds[in_text.numpy().decode()]
        return in_text, tf.convert_to_tensor(pos_tag), out_text

    return lambda x, y: tf.py_function(add_pos_tag, inp=(x, y), Tout=(tf.string, tf.string, tf.string))


def get_final_map_function(
    include_pos_tag: str = DEFAULT_POS_TAG_INCLUDE, teacher_forcing: float = DEFAULT_TEACHER_FORCE, transformer_mode: bool = False
):
    if transformer_mode:
        def map_func(x, p, y):

            inp = (x,y)

            if include_pos_tag == "aux":
                inp[POS_OUTPUT_NAME] = tf.one_hot(p, depth=POS_VOCAB_SIZE)
            elif include_pos_tag == "input":
                inp[POS_INPUT_NAME] = p

            return inp
        func = tf.function(map_func)
        def final_func(x,p,y):
            out = final_func(x,p,y)
            out.set_shape([None])
        return final_func
    def map_func(x, p, y):

        inp = {COMMAND_INPUT_NAME: x}
        out = {ACTION_OUTPUT_NAME: tf.one_hot(y, depth=OUT_VOCAB_SIZE)}

        if include_pos_tag == "aux":
            out[POS_OUTPUT_NAME] = tf.one_hot(p, depth=POS_VOCAB_SIZE)
        elif include_pos_tag == "input":
            inp[POS_INPUT_NAME] = p

        if teacher_forcing:
            inp[ACTION_INPUT_NAME] = y

        return inp, out

    return tf.function(map_func)


@tf.function
def add_sos_eos(in_text, pos_text, out_text):
    in_text = tf.strings.join(["<sos>", in_text, "<eos>"], separator=" ")
    pos_text = tf.strings.join(["<sos>", pos_text, "<eos>"], separator=" ")
    out_text = tf.strings.join(["<sos>", out_text, "<eos>"], separator=" ")
    return in_text, pos_text, out_text


def get_vectorizer():
    in_vectorizer = layers.TextVectorization(
        output_sequence_length=MAX_IN_SEQUENCE_LENGTH, output_mode="int", max_tokens=IN_VOCAB_SIZE, standardize=None
    )
    pos_vectorizer = layers.TextVectorization(
        output_sequence_length=MAX_IN_SEQUENCE_LENGTH, output_mode="int", max_tokens=POS_VOCAB_SIZE, standardize=None
    )
    out_vectorizer = layers.TextVectorization(
        output_sequence_length=MAX_OUT_SEQUENCE_LENGTH, output_mode="int", max_tokens=OUT_VOCAB_SIZE, standardize=None
    )

    in_vectorizer.set_vocabulary(IN_VOCAB_FILE)
    pos_vectorizer.set_vocabulary(POS_VOCAB_FILE)
    out_vectorizer.set_vocabulary(OUT_VOCAB_FILE)

    print("In vocabulary:", {i: x for i, x in enumerate(in_vectorizer.get_vocabulary())})
    print("Pos vocabulary:", {i: x for i, x in enumerate(pos_vectorizer.get_vocabulary())})
    print("Out vocabulary:", {i: x for i, x in enumerate(out_vectorizer.get_vocabulary())})

    def vectorize_text(x, p, y):
        x = in_vectorizer(x)
        p = pos_vectorizer(p)
        y = out_vectorizer(y)

        return x, p, y

    return vectorize_text, in_vectorizer, pos_vectorizer, out_vectorizer


def get_dataset(experiment: SPLITS):
    train, test = load_dataset(experiment, split=["train", "test"])

    # Split dataset into tuples
    train = train.map(lambda x: (x["commands"], x["actions"]), num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(lambda x: (x["commands"], x["actions"]), num_parallel_calls=tf.data.AUTOTUNE)

    add_pos_tag = get_add_pos_tag(DS_POS_FILE)

    # Add pos tag to dataset
    train = train.map(add_pos_tag, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(add_pos_tag, num_parallel_calls=tf.data.AUTOTUNE)

    # Add <sos> and <eos> to dataset
    train = train.map(add_sos_eos, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(add_sos_eos, num_parallel_calls=tf.data.AUTOTUNE)

    vectorize_text, in_vectorizer, pos_vectorizer, out_vectorizer = get_vectorizer()

    # Vectorize dataset
    train = train.map(
        lambda x, p, y: tf.py_function(vectorize_text, inp=(x, p, y), Tout=(tf.int64, tf.int64, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test = test.map(
        lambda x, p, y: tf.py_function(vectorize_text, inp=(x, p, y), Tout=(tf.int64, tf.int64, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print(len(train), len(test))

    return train, test, (in_vectorizer, pos_vectorizer, out_vectorizer)


if __name__ == "__main__":
    train, test = get_dataset("simple")
    for example in train.take(1):
        print(example)
        break
