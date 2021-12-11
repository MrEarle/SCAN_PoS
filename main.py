import os

from comet_ml import Optimizer
from src.tasks import train
from src.data.scan import get_dataset
from src.utils.constants import ACTION_OUTPUT_NAME, get_default_params

if __name__ == "__main__":
    train_ds, test_ds, (in_vec, _, _) = get_dataset("addprim_jump")
    pad_idx = in_vec.get_vocabulary().index("")
    start_idx = in_vec.get_vocabulary().index("<sos>")
    end_idx = in_vec.get_vocabulary().index("<eos>")

    params = {**get_default_params(), "epochs": 40}

    # Optimizer parameters:
    opt_params = {
        # We pick the Bayes algorithm:
        "algorithm": "bayes",
        "name": "SCAN-PoS Hyperparam Optimizer -- Jump Split",
        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "hidden_layers": {"type": "discrete", "values": [1, 2]},
            "hidden_size": {"type": "discrete", "values": [100, 200]},
            "dropout": {"type": "discrete", "values": [0.1, 0.5]},
            "use_attention": {"type": "discrete", "values": [False, True]},
            # "include_pos_tag": {"type": "categorical", "values": ["aux", "input"]},
            "include_pos_tag": {"type": "categorical", "values": [""]},
        },
        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": f"test_action_output_accuracy",
            "objective": "maximize",
        },
    }

    # Comet ML Experiment
    optimizer = Optimizer("b389ea20499849eda68625590eac53ac")

    models = os.listdir("snap")

    for experiment in optimizer.get_experiments(project_name="SCAN-PoS-Split(jump)"):
        params = {
            **params,
            "hidden_size": experiment.get_parameter("hidden_size"),
            "hidden_layers": experiment.get_parameter("hidden_layers"),
            "dropout": experiment.get_parameter("dropout"),
            "use_attention": experiment.get_parameter("use_attention"),
            "include_pos_tag": experiment.get_parameter("include_pos_tag"),
            "batch_size": 512,
        }

        params[
            "name"
        ] = f"h_size({params['hidden_size']})-h_layers({params['hidden_layers']})-dropout({params['dropout']})"

        if params["include_pos_tag"] != "":
            params["name"] += f"-pos({params['include_pos_tag']})"

        if params["use_attention"]:
            params["name"] += "-attention"

        if params["name"] in models:
            continue

        train.train(train_ds, test_ds, pad_idx, start_idx, end_idx, experiment, params)
