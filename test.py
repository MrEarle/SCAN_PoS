from tensorflow.keras import layers
import tensorflow as tf

from src.models.lstm import Seq2SeqAttentionLSTM
from src.data.scan import (
    MAX_SEQUENCE_LENGTH,
    IN_VOCAB_FILE,
    POS_VOCAB_SIZE,
    OUT_VOCAB_SIZE,
    IN_VOCAB_SIZE,
    POS_VOCAB_FILE,
    OUT_VOCAB_FILE,
)
from src.utils.constants import ACTION_INPUT_NAME, ACTION_OUTPUT_NAME, COMMAND_INPUT_NAME, POS_OUTPUT_NAME

# tf.compat.v1.disable_eager_execution()


in_vectorizer = layers.TextVectorization(
    output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode="int", max_tokens=IN_VOCAB_SIZE, standardize=None
)
pos_vectorizer = layers.TextVectorization(
    output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode="int", max_tokens=POS_VOCAB_SIZE, standardize=None
)
out_vectorizer = layers.TextVectorization(
    output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode="int", max_tokens=OUT_VOCAB_SIZE, standardize=None
)


in_vectorizer.set_vocabulary(IN_VOCAB_FILE)
pos_vectorizer.set_vocabulary(POS_VOCAB_FILE)
out_vectorizer.set_vocabulary(OUT_VOCAB_FILE)

out_voc = out_vectorizer.get_vocabulary()
pos_voc = pos_vectorizer.get_vocabulary()

model_in = "<sos> jump and run <eos>"

model_in_vec = in_vectorizer(tf.convert_to_tensor([model_in, model_in]))

print(model_in_vec)
print(model_in_vec.shape)


model = Seq2SeqAttentionLSTM(
    hidden_size=200,
    hidden_layers=1,
    include_pos_tag="aux",
    teacher_forcing=0,
)

pre_res = model({COMMAND_INPUT_NAME: model_in_vec}, training=False)
print(pre_res.keys())

actions = tf.argmax(pre_res[ACTION_OUTPUT_NAME][0], axis=-1)
pos = tf.argmax(pre_res[POS_OUTPUT_NAME][0], axis=-1)

print(actions, pos)

actions = " ".join([out_voc[i] for i in actions]).strip()
pos = " ".join([pos_voc[i] for i in pos]).strip()

print(actions)
print(pos)

checkpoint_path = "snap\h_size(200)-h_layers(1)-dropout(0.1)-pos(aux)/best_action_accuracy"
model.load_weights(checkpoint_path)

pre_res = model({COMMAND_INPUT_NAME: model_in_vec}, training=False)
print(pre_res.keys())

actions = tf.argmax(pre_res[ACTION_OUTPUT_NAME][0], axis=-1)
pos = tf.argmax(pre_res[POS_OUTPUT_NAME][0], axis=-1)

print(actions, pos)

actions = " ".join([out_voc[i] for i in actions]).strip()
pos = " ".join([pos_voc[i] for i in pos]).strip()

print(actions)
print(pos)
