from typing import TypedDict


COMMAND_INPUT_NAME = "input_embedding"
POS_INPUT_NAME = "pos_embedding"
ACTION_INPUT_NAME = "action_onehot"

ACTION_OUTPUT_NAME = "action_output"
POS_OUTPUT_NAME = "pos_output"

DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_SIZE = 25
DEFAULT_HIDDEN_LAYERS = 1
DEFAULT_TEACHER_FORCE = 0.5
DEFAULT_POS_TAG_INCLUDE = ""


class Params(TypedDict):
    dropout: float
    hidden_size: int
    hidden_layers: int
    teacher_forcing: float
    include_pos_tag: str
    epochs: int
    batch_size: int
    name: str
    use_attention: bool


def get_default_params() -> Params:
    return {
        "dropout": DEFAULT_DROPOUT,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "hidden_layers": DEFAULT_HIDDEN_LAYERS,
        "teacher_forcing": DEFAULT_TEACHER_FORCE,
        "include_pos_tag": DEFAULT_POS_TAG_INCLUDE,
        "epochs": 10,
        "batch_size": 512,
        "name": "default",
        "use_attention": False,
    }
