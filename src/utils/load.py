from .constants import COMMAND_INPUT_NAME, POS_INPUT_NAME
import tensorflow as tf


def load_model(model, in_vectorizer, pos_vectorizer, name):
    model_in = "<sos> jump and run <eos>"
    model_pos = "NOUN CONJ VERB"

    model_in_vec = in_vectorizer(tf.convert_to_tensor([model_in, model_in]))
    model_pos_vec = pos_vectorizer(tf.convert_to_tensor([model_pos, model_pos]))

    model.teacher_forcing = False
    model({COMMAND_INPUT_NAME: model_in_vec, POS_INPUT_NAME: model_pos_vec}, training=False)
    checkpoint_path = f"snap\{name}/best_action_accuracy"
    model.load_weights(checkpoint_path)
    model.teacher_forcing = True
    return model
