from src.data.scan import load_dataset
from tqdm import tqdm
import tensorflow as tf

tf.executing_eagerly()

train, test = load_dataset("simple", split=["train", "test"])


def m_l(train, test):
    max_in_len = 0
    max_out_len = 0

    for sentence in tqdm(train):
        max_in_len = max(max_in_len, len(tf.strings.split(sentence["commands"])))
        max_out_len = max(max_out_len, len(tf.strings.split(sentence["actions"])))

    for sentence in tqdm(test):
        max_in_len = max(max_in_len, len(tf.strings.split(sentence["commands"])))
        max_out_len = max(max_out_len, len(tf.strings.split(sentence["actions"])))

    print(max_in_len, max_out_len, flush=True)
    return max_in_len, max_out_len


# m_l = lambda x, y: tf.py_function(m_l, [x, y], [tf.int64, tf.int64])

m_in, m_out = m_l(train, test)

print(m_in, m_out)
print(tf.make_ndarray(m_in), tf.make_ndarray(m_out))
