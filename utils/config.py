import ml_collections

import os
import yaml
import textwrap
from typing import Any, Dict, List, Tuple


def get_dset_config_from_experiment(experiment_dir: str):
    config_dir = os.path.join(experiment_dir, "config")
    dset_config_path = os.path.join(config_dir, "dset_config.yml")
    dset_config = load_yaml_config(dset_config_path)
    return dset_config


def get_model_config_from_experiment(experiment_dir: str):
    config_dir = os.path.join(experiment_dir, "config")
    model_config_path = os.path.join(config_dir, "model_config.yml")
    model_config = load_yaml_config(model_config_path)
    return model_config


def get_weights_path_from_experiment(experiment_dir: str):
    weights_dir = os.path.join(experiment_dir, "weights")
    epoch_weights_paths = os.listdir(weights_dir)

    # Sort the files according to suffix if there are files with
    # '_XXX' as suffix.
    if any("_" in p for p in epoch_weights_paths):
        epoch_weights_paths = sorted(
            [p for p in epoch_weights_paths if "weights" in p],
            key=get_epoch_number_from_path,
        )
    else:
        epoch_weights_paths = sorted(epoch_weights_paths)

    best_weights_filename = os.path.splitext(epoch_weights_paths[-1])[0]
    best_weights_path = os.path.join(weights_dir, best_weights_filename)
    return best_weights_path


def create_run_dir(dir_name: str, root_dir: str = "results") -> str:
    """Creates a directory for the current execution"""
    if os.path.exists(root_dir):
        dir_content = os.listdir(root_dir)
        dir_content = [os.path.join(root_dir, d) for d in dir_content]
        dir_content = [d for d in dir_content if os.path.isdir(d)]
        next_idx = find_next_idx(dir_content)
    else:
        next_idx = "000"
    run_dir_path = os.path.join(root_dir, f"{next_idx}-{dir_name}")
    os.makedirs(run_dir_path)
    return run_dir_path


def get_path_of_directory_with_id(dir_id: str, results_dir: dir = "results"):

    if isinstance(dir_id, int):
        dir_id = f"{dir_id:03d}"

    if results_dir == "":
        raise ValueError("`results_dir` can not be an empty string")

    experiment_names = sorted(os.listdir(results_dir))
    if len(experiment_names) == 0:
        raise ValueError(f"There are no experiments in {results_dir}")

    experiment_ids = [n.split("-")[0] for n in experiment_names]
    if not dir_id in experiment_ids:
        raise ValueError(f"There is no directory with {dir_id} id")
    else:
        idx_of_dir = experiment_ids.index(dir_id)
        experiment_dir_with_id = experiment_names[idx_of_dir]
        return os.path.join(results_dir, experiment_dir_with_id)


def create_dir_suffix(**kwargs) -> None:
    """Utility function to create the suffix of the dataset folder."""
    suffix = []
    for k, v in kwargs.items():
        if isinstance(v, bool) and v:
            str_ = k if k != "tfrecord" else "tf"
            suffix.append(str_)
    return "_".join(suffix)


def save_dict_as_yaml(save_path: str, config: Dict[str, Any]):
    """Saves `config` dictionary as a yaml file to `save_path`. If the
    dictionary has any tuples, they will get converted to lists before
    saving.

    Args:
        save_path (str): Path where the YAML file will be saved.
        config (Dict[str, Any]): Configuration dictionary.
    """

    for k in config.keys():
        if isinstance(config[k], tuple):
            config[k] = list(config[k])

    with open(save_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def load_yaml_config(yaml_config_path: str) -> Dict[str, Any]:
    """Loads a YAML file to a python dictionary.

    Args:
        yaml_config_path (str): path where the YAML is located.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(yaml_config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_dset_config(
    dset_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Returns a tuple of 3 elements with the configuration assets for the
    dataset inside `dset_dir`. The first element of the tuple is a dictionary
    with the paths to the TFRecords, patients and the modalities inside the
    dataset. The second element of the tuple is the feature description for
    the sliced examples and the third element is the feature description for
    the volume examples.

    Args:
        dset_dir (str): path to the directory containing the dataset

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: 3-tuple with
        configuration dictionaries.
    """
    dset_config_path = os.path.join(dset_dir, "dset_config.yml")
    dset_config = load_yaml_config(dset_config_path)
    return dset_config


def find_next_idx(dir_content: List[str]) -> str:
    """Finds the ID for the next experiment.

    Args:
        dir_content (List[str]): Files inside directory.

    Returns:
        str: Formatted experiment id.
    """
    if len(dir_content) > 0:
        idxs = [d.split("/")[-1].split("-")[0][-3:] for d in dir_content]
        idxs = [int(i) for i in idxs if i.isnumeric()]
        next_idx = max(idxs) + 1 if len(idxs) else 0
    else:
        next_idx = 0
    return f"{next_idx:03d}"


def get_epoch_number_from_path(path: str):
    """Function used to sort the weights paths"""
    filename = os.path.splitext(path)[0]
    number = int(filename.split("_")[1])
    return number


def create_experiment_desc(config: ml_collections.ConfigDict) -> str:
    """Creates a title with the kwargs given to name the experiment folder"""

    tags = []
    tags.append(config.dataloader.dataset)
    tags.append(config.dataloader.modalities.replace(",", "_").replace(" ", ""))
    tags.append(config.model.name)
    tags.append("".join([s.capitalize() for s in config.task.split("_")]))
    tags.append(f"{config.base_lr:.0e}lr".replace("-", ""))
    tags.append(f"{config.weight_decay:.0e}wd".replace("-", ""))
    tags.append(f"{config.dataloader.batch_size}bs")
    if "norm_layer" in config.model:
        tags.append(config.model.norm_layer.split(".")[-1])

    return "-".join(tags).replace(".", "")


def save_md_with_experiment_config(
    save_path: str, config: ml_collections.ConfigDict, task: str
) -> str:
    """Create a markdown file with the experiment configuration."""

    def create_items(config: ml_collections.ConfigDict) -> str:
        items = []
        for k, v in config.iteritems():
            if not isinstance(v, ml_collections.ConfigDict):
                items.append(f"- {k}: {v}")
        return "\r".join(items)

    template = f"""\
    # {config.dataloader.dataset} {config.dataloader.modalities} - {task}

    ## Model details:
    ```
    {textwrap.dedent(create_items(config.model))}
    ```

    ## Dataset details:
    ```
    {textwrap.dedent(create_items(config.dataloader))}
    ```

    ## Additional details:
    ```
    {textwrap.dedent(create_items(config))}
    ```
    """
    template = textwrap.dedent(template)

    with open(save_path, "w") as f:
        f.write(template)
