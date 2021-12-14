from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, MeanMetricWrapper
import tensorflow as tf


def get_masked_categorical_crossentropy(mask_value):
    mask_value = K.variable(mask_value)

    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character ''
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)

    return masked_categorical_crossentropy


def get_masked_categorical_accuracy(mask_value):
    mask_value = K.variable(mask_value)

    def masked_categorical_accuracy(y_true, y_pred):
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        acc = categorical_accuracy(y_true, y_pred) * mask

        return K.sum(acc) / K.sum(mask)

    return masked_categorical_accuracy


class MaskedCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, mask_value, name="masked_accuracy", dtype=None):
        maksed_categorical_accuracy = get_masked_categorical_accuracy(mask_value)
        super(MaskedCategoricalAccuracy, self).__init__(maksed_categorical_accuracy, name, dtype=dtype)
