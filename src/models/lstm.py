from abc import abstractmethod
from os import name
from typing import Optional
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from ..data.scan import IN_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, OUT_VOCAB_SIZE


class AbsSeq2SeqLSTM(keras.Model):
    def __init__(self, hiden_dim=128, dropout=0.1, start_idx=0, end_idx=2, pad_idx=1, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        self.in_embedding = layers.Embedding(IN_VOCAB_SIZE, hiden_dim, mask_zero=True, name='input_embedding')
        self.out_embedding = layers.Embedding(OUT_VOCAB_SIZE, OUT_VOCAB_SIZE, mask_zero=True, name='output_embedding')

        self.mask = layers.Masking(mask_value=self.pad_idx, name='mask')

        self.in_forward_lstm = layers.LSTM(units=hiden_dim, return_sequences=True, return_state=True, name='input_forward_lstm')
        self.in_backward_lstm = layers.LSTM(units=hiden_dim, return_sequences=True, return_state=True, name='input_backward_lstm')
        self.out_lstm = layers.LSTMCell(
            units=hiden_dim,
            dropout=dropout,
            name='output_lstm'
        )

        self.softmax_out = layers.Softmax()
        self.linear_out = layers.Dense(units=OUT_VOCAB_SIZE, activation=self.softmax_out, name='output_dense')

    def call(self, inputs, max_length=MAX_SEQUENCE_LENGTH, teacher_actions: Optional[tf.Tensor] = None, training=True):
        batch_size = inputs.shape[0]

        embedded_inputs = self.in_embedding(self.mask(inputs))
        f_all_h, f_fin_h, f_fin_c = self.in_forward_lstm(embedded_inputs, training=training)
        b_all_h, b_fin_h, b_fin_c = self.in_backward_lstm(embedded_inputs, training=training)

        all_hidden = f_all_h + b_all_h
        final_hidden = f_fin_h + b_fin_h
        final_cell = f_fin_c + b_fin_c

        state = (final_hidden, final_cell)

        prediction = tf.repeat(tf.one_hot([self.start_idx], depth=OUT_VOCAB_SIZE), [batch_size], axis=0)

        out = []
        self.out_lstm.reset_dropout_mask()
        self.out_lstm.reset_recurrent_dropout_mask()
        for i in range(max_length):
            if teacher_actions is None:
                y_t = tf.argmax(prediction, axis=-1)
            else:
                y_t = teacher_actions[i, :]

            y_t = self.out_embedding(y_t)

            prediction, state = self.decode_step(y_t, state, all_hidden, training=training)
            out.append(prediction)

        predictions = tf.stack(out, axis=1)

        return predictions

    @abstractmethod
    def decode_step(self, y_t, state, all_hidden, training=True):
        pass


class Seq2SeqLSTM(AbsSeq2SeqLSTM):
    def decode_step(self, y_t, state, all_hidden, training=True):
        x, state = self.out_lstm(y_t, state, training=training)

        prediction = self.linear_out(x)
        return prediction, state


class Seq2SeqAttentionLSTM(AbsSeq2SeqLSTM):
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
