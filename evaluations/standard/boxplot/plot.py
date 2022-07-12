from typing import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


PATH_TO_FIGURE_DIR = './evaluations/standard/boxplot/figures'

def boxplot_overall(overall_means):
    overall_means = sorted(overall_means.items(), key=lambda x: np.median(x[1]))
    gnn_index = list(filter(lambda x: overall_means[x][0] == 'GNN-RL', range(len(overall_means))))[0]
    overall_means = OrderedDict(overall_means)

    df = pd.DataFrame(overall_means.values(), index=overall_means.keys())
    outline = '#1E40AF'
    fig = df.T.boxplot(patch_artist=True, figsize=(6, 7.5), vert=True, color=dict(boxes='#BFDBFE', whiskers=outline, medians=outline, caps=outline), return_type='both')
    plt.subplots_adjust(bottom=0.25)
    # plt.xticks(ticks=[i for i in range(0, 200, 15)], rotation=25)

    red_outline = '#EF4444'
    for i in ['whiskers', 'caps']:
        fig.lines[i][gnn_index * 2].set_color(red_outline)
        fig.lines[i][gnn_index * 2 + 1].set_color(red_outline)
    fig.lines['boxes'][gnn_index].set_color('#FECACA')
    fig.lines['medians'][gnn_index].set_color(red_outline)

    fig.ax.set_title('Overall mean gap to best solution')
    fig.ax.set_xlabel('Mean gap to best solution (%)')
    fig.ax.set_ylabel('Algorithm')
    # plt.show()
    plt.savefig(f'{PATH_TO_FIGURE_DIR}/overall_gap.svg', format="svg")


def boxplot_by_problem_size(results):
    # k = MK01
    # v = { 'spt': [...] }
    for k, v in results.items():
        v = sorted(v.items(), key=lambda x: np.median(x[1]))
        gnn_index = list(filter(lambda x: v[x][0] == 'gnn', range(len(v))))[0]
        v = OrderedDict(v)

        df = pd.DataFrame(v.values(), index=[i.upper() for i in v.keys()])
        blue_outline = '#1E40AF'
        red_outline = '#EF4444'
        bp = df.T.boxplot(
            patch_artist=True,
            figsize=(7.5, 3),
            vert=False,
            return_type='both',
            color=dict(boxes='#BFDBFE', whiskers=blue_outline, medians=blue_outline, caps=blue_outline)
        )
        plt.subplots_adjust(bottom=0.25)
        plt.xticks(rotation=25)

        for i in ['whiskers', 'caps']:
            bp.lines[i][gnn_index * 2].set_color(red_outline)
            bp.lines[i][gnn_index * 2 + 1].set_color(red_outline)

        bp.lines['boxes'][gnn_index].set_color('#FECACA')
        bp.lines['medians'][gnn_index].set_color(red_outline)
        bp.ax.set_title(f'{k}')
        bp.ax.set_xlabel('Gaps to best solution (%)')
        bp.ax.set_ylabel('Algorithm')
        plt.savefig(f'{PATH_TO_FIGURE_DIR}/by_problem_size/{k}.svg', format="svg")
        # plt.show()
        plt.close()

    
def boxplot_by_problem_size_double(results):
    for k, v in results.items():
        v = sorted(v.items(), key=lambda x: np.median(x[1]))
        gnn_index = list(filter(lambda x: v[x][0] == 'gnn', range(len(v))))[0]
        v = OrderedDict(v)

        df = pd.DataFrame(v.values(), index=[i.upper() for i in v.keys()])
        blue_outline = '#1E40AF'
        red_outline = '#EF4444'
        bp = df.T.boxplot(
            patch_artist=True,
            figsize=(7.5, 3),
            vert=True,
            return_type='both',
            color=dict(boxes='#BFDBFE', whiskers=blue_outline, medians=blue_outline, caps=blue_outline)
        )
        plt.subplots_adjust(bottom=0)
        plt.xticks(rotation=25)

        for i in ['whiskers', 'caps']:
            bp.lines[i][gnn_index * 2].set_color(red_outline)
            bp.lines[i][gnn_index * 2 + 1].set_color(red_outline)

        bp.lines['boxes'][gnn_index].set_color('#FECACA')
        bp.lines['medians'][gnn_index].set_color(red_outline)
        bp.ax.set_title(f'Gaps to best solution for {k}')
        bp.ax.set_xlabel('Gaps to best solution (%)')
        bp.ax.set_ylabel('Algorithm')
        plt.savefig(f'{PATH_TO_FIGURE_DIR}/by_problem_size/{k}.svg', format="svg")
        # plt.show()
        plt.close()


def dotplot(overall_means):
    '''
    plt.scatter(x, y)
    x and y are array of length 40 (because there are 40 points)
    x = MK01, MK02, MK03, ..., MK10
    y = [spt, ..., gnn-rl]
    '''
    labels = ['SPT', 'MWKR', 'LWKR', 'GNN-RL']
    x = [i for i in range(1, 11)]
    y = [overall_means[i] for i in labels]
    y = list(itertools.chain(y))
    markers = ['d', 'v', 's', '*']
    # print(overall_means)

    fig, ax = plt.subplots()

    for i in range(len(y)):
        ax.scatter(x, y[i], marker=markers[i], label=labels[i])
    pos = ax.get_position()
    ax.set_xticks([x for x in range(1, 11)])
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlabel('Problem distribution (MK01 to MK10)')
    ax.set_ylabel('Mean gap to best solution (%)')
    ax.set_title('Mean gap to best solution (MK01 to MK10)')
    plt.savefig('evaluations/standard/boxplot/figures/dotplot.svg', format="svg")
    plt.close()



if __name__ == '__main__':
    import json
    import csv

    dispatching_rule_results = json.load(open('./evaluations/standard/boxplot/dispatching_rules_results.json'))
    gnn_results = json.load(open('./evaluations/standard/boxplot/gnn_results.json'))

    # means of 12 results for each problem size (MK01 - MK10)
    spt_means = []
    mwkr_means = []
    lwkr_means = []
    gnn_means = []

    # results for each 12 problems in each problem size (MK01 - MK10)
    results_for_each_problem_size = {}

    for i in range(1, 11):
        problem_name = 'MK{:02d}'.format(i)
        results_for_each_problem_size[problem_name] = {}
        ortools_results = csv.reader(open('./evaluations/standard/boxplot/ortools_solutions/RESULTS_MK{:02d}_12.csv'.format(i)))
        ortools_results = [x[2] for x in list(ortools_results)][1:]
        ortools_results = np.array([float(x) for x in ortools_results], dtype=np.float32)

        result = dispatching_rule_results[problem_name]
        gnn_result = np.array(gnn_results[problem_name], dtype=np.float32)

        spt = (result['spt'] - ortools_results) / ortools_results * 100
        mwkr = (result['mwkr'] - ortools_results) / ortools_results * 100
        lwkr = (result['lwkr'] - ortools_results) / ortools_results * 100
        gnn = (gnn_result - ortools_results) / ortools_results * 100

        results_for_each_problem_size[problem_name]['spt'] = spt.tolist()
        results_for_each_problem_size[problem_name]['mwkr'] = mwkr.tolist()
        results_for_each_problem_size[problem_name]['lwkr'] = lwkr.tolist()
        results_for_each_problem_size[problem_name]['gnn'] = gnn

        spt_means.append(np.mean(spt))
        mwkr_means.append(np.mean(mwkr))
        lwkr_means.append(np.mean(lwkr))
        gnn_means.append(np.mean(gnn))
        
    # mean over all the means of each problem size (MK01 - MK10)
    overall_means = {
        'GNN-RL': gnn_means,
        'SPT': spt_means,
        'MWKR': mwkr_means,
        'LWKR': lwkr_means,
    }

    # boxplot_overall(overall_means)
    boxplot_by_problem_size(results_for_each_problem_size)
    # dotplot(overall_means)
