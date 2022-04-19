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

