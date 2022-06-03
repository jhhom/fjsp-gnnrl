# To train the model, simply run
1. Change the configuration in `params.py`
2. Run `python train.py`


# Configuration
## Problem Size
To change the problem distribution, change this line in `params.py`

```
config.size = 'MK04'
```

`config.size` can only be one of these values:

1. `MK01`
2. `MK02`
3. `MK03`
4. `MK04`
5. `MK05`
6. `MK06`
7. `MK07`
8. `MK08`
9. `MK09`
10. `MK10`

Based on `config.size`, the training problem instances will be generated with the limits specified in `uniform_instance_gen.py`:

```
datasetConfigs = {
    "MK01": ...
}
```

## Choosing the folder to save training logs and trained model
Steps:

1. In `params.py`, change the "ID_1" part of this line. It can be "ID_2" or "ID_3", or any "ID_<any number you like>".
    ```python
    config.progress_config.path_to_save_progress = f'./records/{config.size}/ID_1'
    ```
2. Run `train.py`

The convention of folder naming system is `records/<PROBLEM_SIZE>/ID_<ANY_NUMBER>`

Every 100 training epochs, the folder will save:

* Validation log
* Training log
* Best model weight (based on validation performance)

In saved mode, the folder will additionally store:

* Optimizer weight

## To save training checkpoint
To save training checkpoint so that you can pause and resume training later on, just change this line in `params.py`

```
config.progress_config.save_training = True
```

`save_training` will trigger the saving of optimizer weights every 100 training epochs.

## To resume training from a saved checkpoint
1. Set this as `True`
    ```
    config.progress_config.save_training = True
    ```

2. Set the `path_to_save_progress` in `params.py` to the path of the saved checkpoint
    ```
    config.progress_config.path_to_save_progress = f'./records/{config.size}/ID_1'
    ```