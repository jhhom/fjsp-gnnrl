import numpy as np

from .fjsp_env import FJSP

env = FJSP(3, 3, 3)
problem = np.array(
    [
        [
            [1, 3, 2],
            [0, 2, 1],
            [0, 0, 3],
        ],
        [
            [0, 0, 2],
            [2, 2, 1],
            [0, 0, 0]
        ],
        [
            [2, 1, 0],
            [1, 0, 1],
            [3, 0, 0]
        ]
    ], dtype=np.int32
)


i = 1
def step_and_print(action):
    global i
    print(f'STEP {i}')
    print(f'ACTION - OPERATION {action[0]} | MACHINE {action[1]}')
    adj, fea, reward, done, omega, mask = env.step((action[0] - 1, action[1] - 1))
    print('ADJACENCY')
    print(adj)
    print()
    print('FEATURE')
    print(fea)
    print()
    print(f'REWARD {reward}')
    print(f'DONE {done}')
    print(f'OMEGA {omega}')
    print(f'MASK {mask}')
    print()
    print('----------------------------------------------------')
    print()
    i += 1


env.reset(problem, 3)
step_and_print((1, 2))
step_and_print((2, 3))
step_and_print((4, 3))
step_and_print((3, 3))
step_and_print((5, 1))
step_and_print((6, 1))
step_and_print((7, 1))
step_and_print((8, 1))
print(env.left_shifted_flags)

