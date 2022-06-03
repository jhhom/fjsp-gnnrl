from dataclasses import dataclass

dataclass(frozen=True)
class Characteristic:
    num_of_jobs: int
    num_of_machines: int
    num_of_operations_ub: int

    def __init__(self, n_j, n_m, n_o_ub):
        self.num_of_jobs = n_j
        self.num_of_machines = n_m
        self.num_of_operations_ub = n_o_ub


characteristics = {
    'Mk01.fjs': Characteristic(10, 6, 6),
    'Mk02.fjs': Characteristic(10, 6, 6),
    'Mk03.fjs': Characteristic(15, 8, 10),
    'Mk04.fjs': Characteristic(15, 8, 9),
    'Mk05.fjs': Characteristic(15, 4, 9),
    'Mk06.fjs': Characteristic(10, 15, 15),
    'Mk07.fjs': Characteristic(20, 5, 5),
    'Mk08.fjs': Characteristic(20, 10, 14),
    'Mk09.fjs': Characteristic(20, 10, 14),
    'Mk10.fjs': Characteristic(20, 15, 14),
}