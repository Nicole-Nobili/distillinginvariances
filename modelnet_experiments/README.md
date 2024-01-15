# ModelNet40 Experiments
---

The `configs` folder holds all the configuration files that reproduce every experiment contained in the [Understanding Knowledge Distillation by Transferring Symmetries]().

## Dependencies

Try to set up a conda environment by using the environment file in this directory.
```
conda env create --name [your_env_name] --file=conda_env.yml
```

If this does not work, install the following packages manually:


## Running trainings

To run a training, use, for example
```
python train.py --config configs/config_paper_deepsinv_1000p.yml --device cuda:0
```

This will train an invariant deepsets model on ModelNet40 data downsampled to 1000
points. The hyperparameters of the model and of the training procedure are found in the
respective config file.

## Running distillations
