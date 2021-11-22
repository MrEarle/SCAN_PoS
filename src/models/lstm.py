import tensorflow as tf
from tensorflow.keras import layers

from src.utils.constants import *

from .lstm_base import AbsSeq2SeqLSTM


class Seq2SeqLSTM(AbsSeq2SeqLSTM):
    """Seq2Seq LSTM. Without Attention"""

    def decode_step(self, y_t, state, all_hidden, training=True):
        x, state = self.out_lstm(y_t, state, training=training)

        prediction = self.linear_out(x)
        return prediction, state


class Seq2SeqAttentionLSTM(AbsSeq2SeqLSTM):
    """Attention Seq2Seq."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = layers.Attention(use_scale=True, name="attention")

    def decode_step(self, y_t, state, all_hidden, training=True):
        x, state = self.out_lstm(y_t, state, training=training)

        # x: [batch_size, hiden_dim]
        # all_hidden: [batch_size, max_length, hiden_dim]
        attn = self.attention([tf.expand_dims(x, axis=1), all_hidden])

        concat = tf.concat((x, tf.squeeze(attn)), axis=-1)
        prediction = self.linear_out(concat)

        return prediction, state
