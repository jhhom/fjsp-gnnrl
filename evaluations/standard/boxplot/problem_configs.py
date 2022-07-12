from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetConfig:
    num_of_jobs: int
    num_of_machines: int
    highest_num_of_operations_per_job: int
    lowest_num_of_operations_per_job: int
    num_of_alternative_bounds: 'tuple[int, int]'
    duration_bounds: 'tuple[int, int]'


datasetConfigs = {
    "MK01": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=6,
        duration_bounds=(1, 6),
        num_of_alternative_bounds=(1, 3),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=6,
    ),
    "MK02": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=6,
        duration_bounds=(1, 6),
        num_of_alternative_bounds=(1, 6),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=6,
    ),
    "MK03": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=8,
        duration_bounds=(1, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=10,
    ),
    "MK04": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=8,
        duration_bounds=(1, 9),
        num_of_alternative_bounds=(1, 3),
        lowest_num_of_operations_per_job=3,
        highest_num_of_operations_per_job=9
    ),
    "MK05": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=4,
        duration_bounds=(5, 9),
        num_of_alternative_bounds=(1, 2),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=9,
    ),
    "MK06": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=15,
        duration_bounds=(1, 9),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=15,
        highest_num_of_operations_per_job=15,
    ),
    "MK07": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=5,
        duration_bounds=(1, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=5,
    ),
    "MK08": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=10,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 2),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
    ),
    "MK09": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=10,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
    ),
    "MK10": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=15,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
    ),
}
