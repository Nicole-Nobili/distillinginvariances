# Run the distillation framework.
# Train a student network through the knowledge distillation framework.

import torch.nn as nn
from torch.nn.functional import softmax
import torch
import numpy as np

from distiller import Distiller
import util


def main(config: dict):
    device = util.define_torch_device()
    outdir = util.make_output_directory("distilled_deepsets", config["outdir"])
    util.save_config_file(config, outdir)

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory("distilled_deepsets", config["outdir"])
