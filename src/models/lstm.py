import tensorflow as tf
from tensorflow.keras import layers

from .lstm_base import BASE_SEQ2SEQ_LSTM


class Seq2SeqLSTM(BASE_SEQ2SEQ_LSTM):
    """Seq2Seq LSTM. Without Attention"""
    def decode_step(self, y_t, state, all_hidden, training=True):
        x, state = self.out_lstm(y_t, state, training=training)

        prediction = self.linear_out(x)
        return prediction, state


class Seq2SeqAttentionLSTM(BASE_SEQ2SEQ_LSTM):
    """Attention Seq2Seq."""
    def __init__(self, hiden_dim=128, dropout=0.1, start_idx=0, end_idx=2, pad_idx=1, **kwargs):
        super().__init__(
            hiden_dim=hiden_dim,
            dropout=dropout,
            start_idx=start_idx,
            end_idx=end_idx,
            pad_idx=pad_idx,
            **kwargs
        )

        self.attention = layers.Attention(use_scale=True, name='attention')

    def decode_step(self, y_t, state, all_hidden, training=True):
        x, state = self.out_lstm(y_t, state, training=training)

        # x: [batch_size, hiden_dim]
        # all_hidden: [batch_size, max_length, hiden_dim]
        attn = self.attention([tf.expand_dims(x, axis=1), all_hidden])

        concat = tf.concat((x, tf.squeeze(attn)), axis=-1)
        prediction = self.linear_out(concat)

        return prediction, state
