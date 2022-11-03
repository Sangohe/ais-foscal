# Datasets
To validate our method, we trained on three public datasets named ISLES2017, ATLASv1 and ATLASv2. To make things easier, we implemented a CLI that creates TFRecords for training, validation and testing.

# Usage
Run from the project directory


Or change directory to the datasets utils and execute

The CLI commands can be executed from the project directory

```
python -m utils.datasets.tools <command>
```

Or by changing directory to `utils/datasets`. 

```
cd utils/datasets/
python tools.py <command>
```

## Commands
The CLI comes with the following command list. Note that the commands listed are called from within the `utils/datasets` directory. Adjust the command as suggested above if you want to run them from the project directory.

### Help
Help command shows you how to use the CLI. Additionally, it returns a list of commands available. To invoke the help command run:

```
python tools.py --help
```

Also, the help command is available for the other commands. To invoke help for other commands use:

```
python tools.py <command> --help
```

### ISLES2017
#### Train-val-test 
This command expects the path to the ISLES2017 dataset to create train, validation and test partititons. To do this, run the following command:

```
python tools.py isles2017-train-val-test \
    --source-dset-dir=<path-to-isles2017-dataset> \
    --slices --min-max-norm
```
### ATLASv1
#### Train-val
This command expects the path to the ATLASv1 dataset to create the train and validation partititons. To do this, run the following command:

```
python tools.py atlasv1-train-val \
    --source-dset-dir=<path-to-atlasv1-dataset> \
    --slices --min-max-norm
```