# To train the model, simply run

```
python train.py
```

# Changing problem type
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

Based on `config.size`, the training problem instances will be generated with the limits specified in `uniform_instance_gen.py`

```
datasetConfigs = {
    "MK01": ...
}
```

# Choosing the file to save training progress
Steps:

1. In `params.py`, change the "ID_1" part of this line. It can be "ID_2" or "ID_3", or any "ID_<any number you like>".
    ```python
    config.progress_config.path_to_save_progress = f'./records/{config.size}/ID_1'
    ```
2. Run `train.py`

The convention of folder naming system is `records/<PROBLEM_SIZE>/ID_<ANY_NUMBER>`


The folder will store:

* Validation log
* Training log
* Best weight

In saved mode, the folder will additionally store:







