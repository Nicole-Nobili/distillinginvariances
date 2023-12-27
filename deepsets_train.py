#!/usr/bin/env python

# Run the training of the deepsets network given the hyperparameters listed below.
import argparse
import os
import sys

sys.path.append("..")

from mlp.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--gpu", type=str, default="", help="Sets the number of the GPU to run on."
)
parser.add_argument(
    "--test_kfold", type=int, default=-1, help="Which kfold to use for test."
)
parser.add_argument(
    "--nbits",
    type=int,
    default=-1,
    help="The number of bits for quantisation of the model.",
)
parser.add_argument(
    "--outdir", type=str, default="test", help="The output directory name."
)
args = parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Import the data.
kfolds = 5
nconst = 8

data_folder_path = "../../ds_data"
jets_path = f"jets_{nconst}constituents_ptetaphi_robust_fast"

train_kfolds = [kfold for kfold in range(kfolds) if kfold != args.test_kfold]
test_kfold = args.test_kfold
data_hyperparams = {
    "fpath": os.path.join(data_folder_path, jets_path),
    "fnames_train": [
        f"jet_images_c{nconst}_minpt2.0_ptetaphi_robust_fast_{train_kfold}"
        for train_kfold in train_kfolds
    ],
    "fname_test": f"jet_images_c{nconst}_minpt2.0_ptetaphi_robust_fast_{test_kfold}",
}

training_hyperparams = {"epochs": 300, "batch": 128, "lr": 0.0013, "pruning_rate": 0}

compilation_hyperparams = {
    "optimizer": "adam",
    "loss": "softmax_with_crossentropy",
    # "loss":      "categorical_crossentropy",
    "metrics": ["categorical_accuracy"],
}

model_hyperparams = {
    "layers": [120, 60, 32, 64, 64, 64, 32, 44],
    "activ": "relu",
    "l1_coeff": 0.0000131,
    "nbits": args.nbits,
}

args = {
    "data_hyperparams": data_hyperparams,
    "training_hyperparams": training_hyperparams,
    "model_hyperparams": model_hyperparams,
    "compilation": compilation_hyperparams,
    "outdir": args.outdir,
    "mlp_type": "smlp_reg",
}

main(args)
