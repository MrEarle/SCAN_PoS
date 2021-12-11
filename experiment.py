from argparse import ArgumentParser

from comet_ml import Experiment
from src.tasks import train
from src.data.scan import get_dataset
from src.utils.constants import get_default_params

argparser = ArgumentParser()
argparser.add_argument("--hidden_layers", type=int, default=1)
argparser.add_argument("--hidden_size", type=int, default=100)
argparser.add_argument("--dropout", type=float, default=0.1)
argparser.add_argument("--use_attention", action="store_true")
argparser.add_argument("--include_pos_tag", type=str, choices=["input", "aux"], default="")


def get_args():
    return argparser.parse_args()


if __name__ == "__main__":
    train_ds, test_ds, (in_vec, _, _) = get_dataset("addprim_jump")
    pad_idx = in_vec.get_vocabulary().index("")
    start_idx = in_vec.get_vocabulary().index("<sos>")
    end_idx = in_vec.get_vocabulary().index("<eos>")

    args = get_args()

    params = {
        **get_default_params(),
        "epochs": 60,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "use_attention": args.use_attention,
        "include_pos_tag": args.include_pos_tag,
        "batch_size": 512,
    }

    params["name"] = f"h_size({params['hidden_size']})-h_layers({params['hidden_layers']})-dropout({params['dropout']})"

    if params["include_pos_tag"] != "":
        params["name"] += f"-pos({params['include_pos_tag']})"

    if params["use_attention"]:
        params["name"] += "-attention"

    experiment = Experiment(project_name="scan-addprim-jump")

    train.train(train_ds, test_ds, pad_idx, start_idx, end_idx, experiment, params)
