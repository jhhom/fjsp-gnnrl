import matplotlib.pyplot as plt
import numpy as np
'''
mk10 has 20 jobs
colors

1. black        white
2. gray         white
3. gainsboro    black
4. lightcoral   black
5. maroon       white
6. red          white
7. saddlebrown  white
8. darkorange   white
9. darkgoldenrod    white
10. khaki   black
11. darkkhaki   white
12. olive   white
13. darkolivegreen  white
14. greenyellow     black
15. springgreen     black
16. aquamarine      white
17. cadetblue       white
18. dodgerblue      white
19. indigo          white
20. mediumvioletred white



'''

colors = [
    ('#8B5CF6', 'white'),
    ('gray', 'white'),
    ('gainsboro', 'black'),
    ('lightcoral', 'black'),
    ('maroon', 'white'),
    ('red', 'white'),
    ('saddlebrown', 'white'),
    ('darkorange', 'white'),
    ('darkgoldenrod', 'white'),
    ('khaki', 'black'),
    ('darkkhaki', 'white'),
    ('olive', 'white'),
    ('darkolivegreen', 'white'),
    ('greenyellow', 'black'),
    ('springgreen', 'black'),
    ('aquamarine', 'white'),
    ('cadetblue', 'white'),
    ('dodgerblue', 'white'),
    ('indigo', 'white'),
    ('mediumvioletred', 'white')
]

def calculate_horizon(jobs):
    return np.sum(np.amax(jobs, axis=2))

def calculate_horizon_with_arrival_times(jobs, release_times):
    return np.sum(np.amax(jobs, axis=2)) + release_times[-1]

def get_machine_op_durations(machine_op_ids, jobs, op_id_to_job_info):
    machine_op_durations = np.copy(machine_op_ids)
    for i in range(len(machine_op_durations)):
        for j in range(len(machine_op_durations[i])):
            if machine_op_durations[i][j] < 0:
                break
            row, col = op_id_to_job_info[int(machine_op_ids[i][j])]
            machine_op_durations[i][j] = jobs[row][col][i]
    return machine_op_durations


def draw_gantt_chart(
    horizon: int,
    machine_start_times,
    machine_op_durations,
    machine_op_ids,
    op_id_to_job_info,
    n_j,
):
    n_m = len(machine_start_times)
    fig, gnt = plt.subplots()

    fig.set_size_inches(40, 10.5)
    gnt.set_ylim(0, n_m * 10 + 20)
    gnt.set_xlim(0, horizon)

    gnt.set_xlabel('Time')
    gnt.set_ylabel('Machine')

    gnt.set_yticks([5 + 10 * i for i in range(1, n_m+1)])
    gnt.set_yticklabels([f'{i}' for i in range(1, n_m+1)])

    gnt.grid(True)

    # get all the colors used
    gap = int(len(colors) / n_j)
    colors_used = []
    for i in range(n_j):
        colors_used.append(colors[i * gap])

    for i, start_times in enumerate(machine_start_times):
        bars = []
        op_colors = []
        for j, start_time in enumerate(start_times):
            op_id = int(machine_op_ids[i][j])
            job, op = op_id_to_job_info[op_id]
            op_colors.append(colors_used[job][0])

            if start_time < 0:
                break
            width = machine_op_durations[i][j]
            bars.append((start_time, width))
        gnt.broken_barh(bars, ((i+1) * 10, 9), facecolors=op_colors)
    

    # annotate the bars
    for i, start_times in enumerate(machine_start_times):
        for j, start_time in enumerate(start_times):
            x = start_time + (machine_op_durations[i][j] / 2)
            y = i * 10 + 15
            op_id = int(machine_op_ids[i][j])
            job, op = op_id_to_job_info[op_id]
            text_color = colors_used[job][1]

            gnt.text(x, y, f"{job+1}{op+1}", ha='center', va='center', color=text_color)

    plt.show()
    





if __name__ == '__main__':
    import numpy as np

    machine_op_durations = np.array([
        [ 1., -6., -6., -6., -6., -6., -6., -6., -6.],
        [-6., -6., -6., -6., -6., -6., -6., -6., -6.],
        [ 3.,  2., -6., -6., -6., -6., -6., -6., -6.]
    ])
    machine_start_times = np.array([
        [ 3., -6., -6., -6., -6., -6., -6., -6., -6.],
        [-6., -6., -6., -6., -6., -6., -6., -6., -6.],
        [ 0.,  3., -6., -6., -6., -6., -6., -6., -6.]
    ])
    machine_op_ids = np.array([
        [ 1., -3., -3., -3., -3., -3., -3., -3., -3.],
        [-3., -3., -3., -3., -3., -3., -3., -3., -3.],
        [ 0.,  2., -3., -3., -3., -3., -3., -3., -3.]
    ])
    horizon = 16

    op_id_to_job_info = np.array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2]], dtype=np.int32)

    draw_gantt_chart(
        horizon=16,
        machine_start_times=machine_start_times,
        machine_op_durations=machine_op_durations,
        machine_op_ids=machine_op_ids,
        op_id_to_job_info=op_id_to_job_info,
        n_j=3
    )