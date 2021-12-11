from abc import abstractmethod
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

from ..data.scan import IN_VOCAB_SIZE, MAX_OUT_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, OUT_VOCAB_SIZE, POS_VOCAB_SIZE
from ..utils.constants import *


class AbsSeq2SeqLSTM(keras.Model):
    """Base LSTM Seq2Seq model. Works with no pos tag, pos tag as input or pos tag as auxiliary function."""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        teacher_forcing: float = DEFAULT_TEACHER_FORCE,
        include_pos_tag: str = DEFAULT_POS_TAG_INCLUDE,
        start_idx=0,
        end_idx=2,
        pad_idx=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_size
        self.dropout = dropout
        self.num_layers = hidden_layers
        self.teacher_forcing = teacher_forcing
        self.include_pos_tag = include_pos_tag

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        self.in_embedding = layers.Embedding(IN_VOCAB_SIZE, self.hidden_dim, mask_zero=True, name=COMMAND_INPUT_NAME)
        self.out_embedding = layers.Embedding(OUT_VOCAB_SIZE, OUT_VOCAB_SIZE, mask_zero=True, name=ACTION_OUTPUT_NAME)

        if include_pos_tag == "input":
            self.pos_embedding = layers.Embedding(POS_VOCAB_SIZE, self.hidden_dim, mask_zero=True, name=POS_INPUT_NAME)

        self.in_dropout = layers.Dropout(self.dropout)

        self.in_lstm = layers.Bidirectional(
            layers.LSTM(
                self.hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout,
                name="input_lstm",
            ),
            merge_mode="sum",
        )
        if self.num_layers == 2:
            self.in_lstm2 = layers.Bidirectional(
                layers.LSTM(
                    self.hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout,
                    name="input_lstm2",
                ),
                merge_mode="sum",
            )

        # TODO: Remove recurrent dropout
        self.out_lstm = layers.LSTMCell(
            units=self.hidden_dim, recurrent_dropout=self.dropout, dropout=self.dropout, name="output_lstm"
        )

        self.softmax_out = layers.Softmax()
        self.linear_out = layers.Dense(units=OUT_VOCAB_SIZE, activation=self.softmax_out, name="output_dense")

        if include_pos_tag == "aux":
            self.pos_tag_softmax = layers.Softmax()
            self.pos_tag_linear = layers.Dense(POS_VOCAB_SIZE, activation=self.pos_tag_softmax, name=POS_OUTPUT_NAME)

    def call(self, inputs, max_length=MAX_OUT_SEQUENCE_LENGTH, training=True):
        teacher_actions = inputs[ACTION_INPUT_NAME] if self.teacher_forcing else None

        all_hidden, final_hidden, final_cell = self.encode(inputs, training)

        state = (final_hidden, final_cell)
        predictions = self.decode(all_hidden, state, max_length, teacher_actions, training)

        if self.include_pos_tag == "aux":
            pos_softax = self.pos_tag_linear(all_hidden)
            return {ACTION_OUTPUT_NAME: predictions, POS_OUTPUT_NAME: pos_softax}

        return {ACTION_OUTPUT_NAME: predictions}

    def encode(self, inputs, training):
        """Encode inputs."""
        embedded_inputs = self.in_embedding(inputs[COMMAND_INPUT_NAME])

        if self.include_pos_tag == "input":
            embedded_pos = self.pos_embedding(inputs[POS_INPUT_NAME])
            embedded_inputs = tf.concat((embedded_inputs, embedded_pos), axis=-1)

        embedded_inputs = self.in_dropout(embedded_inputs, training=training)

        all_hidden, h_f, o_f, h_b, o_b = self.in_lstm(embedded_inputs)
        final_hidden = h_f + h_b
        final_cell = o_f + o_b

        if self.num_layers == 2:
            all_hidden, h_f, o_f, h_b, o_b = self.in_lstm2(all_hidden)
            final_hidden = h_f + h_b
            final_cell = o_f + o_b

        return all_hidden, final_hidden, final_cell

    def decode(self, all_hidden, state, max_length, teacher_actions, training):
        """Decode sequence"""
        batch_size = all_hidden.shape[0]

        # First token (<sos>)
        prediction = tf.repeat(tf.one_hot([self.start_idx], depth=OUT_VOCAB_SIZE), [batch_size], axis=0)

        if training and self.teacher_forcing:
            teacher_forcing_indices = tf.where(np.random.rand(batch_size) < self.teacher_forcing)

        out = []
        # Por alguna razon sin esto se cae
        self.out_lstm.reset_dropout_mask()
        self.out_lstm.reset_recurrent_dropout_mask()
        for i in range(max_length):
            y_t = tf.argmax(prediction, axis=1)

            if training and self.teacher_forcing:
                # Teacher actions es cuando lo queremos entrenar "haciendo como
                # que elige la acción correcta"
                teacher_actions = tf.gather_nd(teacher_actions, teacher_forcing_indices)
                y_t = tf.tensor_scatter_nd_update(y_t, teacher_forcing_indices, teacher_actions[:, i])

            # Codificar la acción de input
            y_t = self.out_embedding(y_t)

            # Decodificación
            prediction, state = self.decode_step(y_t, state, all_hidden, training=training)
            out.append(prediction)

        predictions = tf.stack(out, axis=1)
        return predictions

    @abstractmethod
    def decode_step(self, y_t, state, all_hidden, training=True):
        pass
