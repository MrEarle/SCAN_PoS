from abc import abstractmethod
from typing import Optional
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from ..data.scan import IN_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, OUT_VOCAB_SIZE, POS_VOCAB_SIZE
from ..utils.args import args
from ..utils.constants import *


class AbsSeq2SeqLSTM(keras.Model):
    """Base LSTM Seq2Seq model. Works with no pos tag or pos tag as input."""
    def __init__(self, hiden_dim=128, dropout=0.1, start_idx=0, end_idx=2, pad_idx=1, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        self.in_embedding = layers.Embedding(IN_VOCAB_SIZE, hiden_dim, mask_zero=True, name=COMMAND_INPUT_NAME)
        self.out_embedding = layers.Embedding(OUT_VOCAB_SIZE, OUT_VOCAB_SIZE, mask_zero=True, name=ACTION_OUTPUT_NAME)

        self.in_forward_lstm = layers.LSTM(units=hiden_dim, return_sequences=True, return_state=True, name='input_forward_lstm')
        self.in_backward_lstm = layers.LSTM(units=hiden_dim, return_sequences=True, return_state=True, go_backwards=True, name='input_backward_lstm')
        self.out_lstm = layers.LSTMCell(
            units=hiden_dim,
            dropout=dropout,
            name='output_lstm'
        )

        self.softmax_out = layers.Softmax()
        self.linear_out = layers.Dense(units=OUT_VOCAB_SIZE, activation=self.softmax_out, name='output_dense')

        if args.include_pos_tag == 'aux':
            self.pos_tag_softmax = layers.Softmax()
            self.pos_tag_linear = layers.Dense(POS_VOCAB_SIZE, activation=self.pos_tag_softmax, name=POS_OUTPUT_NAME)
            

    def call(self, inputs, max_length=MAX_SEQUENCE_LENGTH, teacher_actions: Optional[tf.Tensor] = None, training=True):
        command = inputs[COMMAND_INPUT_NAME]
        all_hidden, final_hidden, final_cell = self.encode(command, training)

        state = (final_hidden, final_cell)
        predictions = self.decode(all_hidden, state, max_length, teacher_actions, training)

        if args.include_pos_tag == 'aux':
            pos_softax = self.pos_tag_linear(all_hidden)
            return {ACTION_OUTPUT_NAME: predictions, POS_OUTPUT_NAME: pos_softax }

        return {ACTION_OUTPUT_NAME: predictions}

    def encode(self, inputs, training):
        """Encode inputs. Doesn't support pos tags as inputs."""
        embedded_inputs = self.in_embedding(inputs)
        f_all_h, f_fin_h, f_fin_c = self.in_forward_lstm(embedded_inputs, training=training)
        b_all_h, b_fin_h, b_fin_c = self.in_backward_lstm(embedded_inputs, training=training)

        all_hidden = f_all_h + b_all_h
        final_hidden = f_fin_h + b_fin_h
        final_cell = f_fin_c + b_fin_c

        return all_hidden, final_hidden, final_cell

    def decode(self, all_hidden, state, max_length, teacher_actions, training):
        """Decode sequence"""
        batch_size = all_hidden.shape[0]

        # First token (<sos>)
        prediction = tf.repeat(tf.one_hot([self.start_idx], depth=OUT_VOCAB_SIZE), [batch_size], axis=0)

        out = []
        # Por alguna razon sin esto se cae
        self.out_lstm.reset_dropout_mask()
        self.out_lstm.reset_recurrent_dropout_mask()
        for i in range(max_length):
            if teacher_actions is not None:
                # Teacher actions es cuando lo queremos entrenar "haciendo como 
                # que elige la acción correcta"
                y_t = teacher_actions[i, :]
            else:
                # Elegir la acción más probable en base a su predicción anterior
                y_t = tf.argmax(prediction, axis=-1)

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

class AbsSeq2SeqLSTMPosInput(AbsSeq2SeqLSTM):
    """Base Seq2Seq LSTM model. Requires pos tags as input."""
    def __init__(self, hiden_dim=128, dropout=0.1, start_idx=0, end_idx=2, pad_idx=1, **kwargs):
        super().__init__(hiden_dim=hiden_dim, dropout=dropout, start_idx=start_idx, end_idx=end_idx, pad_idx=pad_idx, **kwargs)
        self.pos_embedding = layers.Embedding(POS_VOCAB_SIZE, hiden_dim, mask_zero=True, name=POS_INPUT_NAME)

    def encode(self, command, pos, training):
        """Encode inputs. Requires pos tags as inputs."""
        embedded_command = self.in_embedding(command)
        embedded_pos = self.pos_embedding(pos)
        embedded_inputs = tf.concat((embedded_command, embedded_pos), axis=-1)
        f_all_h, f_fin_h, f_fin_c = self.in_forward_lstm(embedded_inputs, training=training)
        b_all_h, b_fin_h, b_fin_c = self.in_backward_lstm(embedded_inputs, training=training)

        all_hidden = f_all_h + b_all_h
        final_hidden = f_fin_h + b_fin_h
        final_cell = f_fin_c + b_fin_c

        return all_hidden, final_hidden, final_cell

                    
    def call(self, inputs, max_length=MAX_SEQUENCE_LENGTH, teacher_actions: Optional[tf.Tensor] = None, training=True):
        command = inputs[COMMAND_INPUT_NAME]
        pos = inputs[POS_INPUT_NAME]

        all_hidden, final_hidden, final_cell = self.encode(command, pos, training)

        state = (final_hidden, final_cell)
        predictions = self.decode(all_hidden, state, max_length, teacher_actions, training)

        return {ACTION_OUTPUT_NAME: predictions}


# Choose Base model depending if we want to include pos tag as input or not
BASE_SEQ2SEQ_LSTM = AbsSeq2SeqLSTMPosInput if args.include_pos_tag == "input" else AbsSeq2SeqLSTM