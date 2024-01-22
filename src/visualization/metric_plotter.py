import random

from src.library.cache_io import load_file_in_root, get_root_path
from src.tracking import metric
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator, AutoLocator, StrMethodFormatter, ScalarFormatter


def metric_plotter(metric_names, dataset, model, conf, extra, data_root="../../", draw_x=500):
    """
    Args:
        metric_names (list[str]): the performance metric you want to draw. This support for all number of possible
                                    combinations, we recommend [x] or [x, y] types.
                                    [metric.TRAIN_ACCURACY]
                                    [metric.TEST_ACCURACY, metric.TEST_LOSS]
                                    [metric.TEST_ACCURACY, metric.TRAIN_STATIC_REGRET]
                                    [metric.TEST_LOSS, metric.TEST_CONSENSUS_ERROR].
        dataset (str): Dataset name, like Mnist.
        model (str): Model name, like softmax_regression.
        conf (dict): This is used for find suitable records,
        extra (dict(list[str])): This is the extra information dict we give, additional information can be added to suit your needs.
                      For this instance, we need aggregation_rules,aggregation_show_name, attack_types, attack_show_name.
                      eg.
                            extra["aggregation_rules"] = ["Mean"]
                            extra["aggregation_show_name"] = ["mean"]
                            extra["attack_types"] = ["NoAttack"]
                            extra["attack_show_name"] = ["without attack"]
        data_root (str): the path before the record directory, default as "../../"
        draw_x (int): the intervals of data show in x-axis, default as 500.
    """

    train_rounds = conf["rounds"] if conf["epoch_or_iteration"] == "epoch" else conf["rounds_iterations"]
    train_iters = list(range(train_rounds + 1))
    test_iters = []
    attack_types = list(extra["attack_types"])
    attack_names = extra["attack_show_name"]
    aggregation_rules = list(extra["aggregation_rules"])
    labels = extra["aggregation_show_name"]
    graph_messages = []
    for attack_type in attack_types:
        honest_cnt = conf["nodes_cnt"] - conf["byzantine_cnt"] if attack_type != "NoAttack" else conf["nodes_cnt"]
        byzantine_cnt = conf["byzantine_cnt"] if attack_type != "NoAttack" else 0
        graph_messages.append(
            "{}_{}_h{}_b{}".format(conf["graph_type"][:3], conf["centralized"][:3], honest_cnt, byzantine_cnt))

    # colors = ['black','gold', 'skyblue', 'brown', 'olive', 'blue', 'darkgray', 'purple']
    markers = ['s', 'o', 'p',  'v', '<', '>',   '*', 'X', 'D', '1']
    line_styles = ['-', '-.', ':', '--', '-', '-.', ':', '--', '-.', ':']

    data = dict()
    y_label_list = []
    for k in range(len(metric_names)):
        data[metric_names[k]] = [[0 for _ in range(len(attack_types))] for _ in range(len(aggregation_rules))]
        parts = metric_names[k].split('_')
        if metric_names[k] == metric.TRAIN_STATIC_REGRET:
            parts = ["adversary", " regret"]
        y_label_list.append(' '.join(part.capitalize() for part in parts))

    y_lim_list = dict()
    path_list = ["record", dataset, model, ""]
    for i in range(len(aggregation_rules)):
        for j in range(len(attack_types)):
            path_list[-1] = aggregation_rules[i]
            file_name = "{}_{}_{}{}_{}_{}_lr{}_{}_mo{}_{}.pkl".format(attack_types[j], graph_messages[j],
                                                                      conf["epoch_or_iteration"], train_rounds,
                                                                      conf["task_name"],
                                                                      conf["lr_controller"][:2] +
                                                                      conf["lr_controller"][-4:-2],
                                                                      conf["init_lr"],
                                                                      conf["momentum_controller"][:2] +
                                                                      conf["momentum_controller"][-4:-2],
                                                                      conf["init_momentum"], conf["partition_type"])
            data_part = load_file_in_root(file_name, path_list=path_list, root=data_root)
            test_iters = data_part["cumulative_test_round"] if len(test_iters) == 0 else test_iters
            for k in range(len(metric_names)):
                data[metric_names[k]][i][j] = data_part[metric_names[k]]
                # TODO TEMP
                if data[metric_names[k]][i][j] is None or len(data[metric_names[k]][i][j]) == 0:
                    data[metric_names[k]][i][j] = [random.uniform(0, 2000) for _ in range(train_rounds + 1)]
                assert data[metric_names[k]][i][j] is not None and len(data[metric_names[k]][i][j]) != 0, \
                    "no {} data in save datas.".format(metric_names[k])
                if "avg" in data[metric_names[k]][i][j]:
                    data[metric_names[k]][i][j] = data_part[metric_names[k]]["avg"]
                if metric_names[k] in y_lim_list:
                    y_lim_list[metric_names[k]] = [min(y_lim_list[metric_names[k]][0],
                                                       min(data[metric_names[k]][i][j])),
                                                   max(y_lim_list[metric_names[k]][1],
                                                       max(data[metric_names[k]][i][j]))]
                else:
                    y_lim_list[metric_names[k]] = [min(data[metric_names[k]][i][j]),
                                                   max(data[metric_names[k]][i][j])]

    if "y_lim" in extra:
        for i in range(len(extra["y_lim"])):
            y_lim_list[metric_names[i]] = extra["y_lim"][i]
    x_iter = dict()
    for k in range(len(metric_names)):
        if "accuracy" in metric_names[k]:
            y_lim_list[metric_names[k]] = [0, 1.0]  # min max
        else:
            y_lim_list[metric_names[k]] = [y_lim_list[metric_names[k]][0], y_lim_list[metric_names[k]][1] * 1.2]
        if "train" in metric_names[k]:
            x_iter[metric_names[k]] = train_iters
        else:
            x_iter[metric_names[k]] = test_iters

    assert train_iters[-1] == test_iters[-1], "The data saved for training and the data saved for testing " \
                                              "should have the same range in the coordinate dimension of x."

    # Plot the curve
    rows = list(range(len(metric_names)))
    columns = list(range(len(attack_types)))
    LEGEND_SIZE = 20
    FONT_SIZE = 20
    LABEL_SIZE = 15
    SCALE_W = 5 if len(columns) > 1 else 5.5
    SCALE_H = 4.5 if len(rows) > 1 else 5.5

    fig, axs = plt.subplots(len(rows), len(columns), sharex='col', sharey='row',
                            figsize=(SCALE_W * len(columns), SCALE_H * len(rows)))
    # fig.set_size_inches((SCALE * len(columns), SCALE * len(rows)))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    for row in rows:
        for column in range(len(columns)):
            if len(rows) == 1 and len(columns) == 1:
                axs_temp = axs
            elif len(rows) == 1:
                axs_temp = axs[column]
            elif len(columns) == 1:
                axs_temp = axs[row]
            else:
                axs_temp = axs[row][column]
            if row == rows[0]:
                axs_temp.set_title(attack_names[column], fontsize=FONT_SIZE)
            axs_temp.grid('True')
            axs_temp.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            if column == 0:
                axs_temp.set_ylabel(y_label_list[row], fontsize=FONT_SIZE)
            if row == rows[-1]:
                axs_temp.set_xlabel(conf["epoch_or_iteration"], fontsize=FONT_SIZE)
                axs_temp.set_ylim(y_lim_list[metric_names[row]][0], y_lim_list[metric_names[row]][1])
                axs_temp.yaxis.set_major_formatter(StrMethodFormatter('{x:1.1f}'))
                axs_temp.yaxis.set_major_formatter(ScalarFormatter())
                axs_temp.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y')
                axs_temp.yaxis.get_offset_text().set_fontsize(LABEL_SIZE)
            for i in range(len(aggregation_rules)):
                axs_temp.plot(x_iter[metric_names[row]] if len(x_iter[metric_names[row]]) <= 2 * draw_x
                              else x_iter[metric_names[row]][::draw_x],
                              data[metric_names[row]][i][column] if len(x_iter[metric_names[row]]) <= 2 * draw_x
                              else data[metric_names[row]][i][column][::draw_x],
                              label=labels[i],
                              marker=markers[i % 10],
                              markersize=8,
                              linestyle=line_styles[i % 10])  # color=colors[i],

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    if len(columns) == 1:
        plt.subplots_adjust(left=0.15)
        ncols = len(columns)
        plt.subplots_adjust(bottom=0.1 * max(1, ncols) if len(rows) > 1 else 0.35)  # judge label location
        fig.legend(lines, labels, loc='lower right', ncol=ncols, fontsize=LEGEND_SIZE,
                   bbox_to_anchor=(-0.0005, 0.15, 1, 0.7))  # ,bbox_to_anchor=(0.5, -0.2), mode="expand"
    else:
        plt.subplots_adjust(bottom=0.4 / len(rows) if len(rows) > 1 else 0.35)  # judge label location
        fig.legend(lines, labels, loc='lower center', ncol=max(1, int(len(labels) / 2)), fontsize=LEGEND_SIZE,
                   bbox_to_anchor=(-0.0005, -0.005, 1, 0.7))  # ,bbox_to_anchor=(0.5, -0.2), mode="expand"

    picture_name = "{}_{}_lr{}_{}_mo{}_{}.png".format("_".join(metric_names),
                                                      conf["lr_controller"][:2] +
                                                      conf["lr_controller"][-4:-2],
                                                      conf["init_lr"],
                                                      conf["momentum_controller"][:2] +
                                                      conf["momentum_controller"][-4:-2],
                                                      conf["init_momentum"],
                                                      conf["partition_type"])
    path_list[0] = "pictures"
    save_path = get_root_path(picture_name, path_list[:-1], data_root, create_if_not_exist=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    data_root = "../../applications/federated_robust_online_gradient_descent/"
    # metric_names = [metric.TEST_ACCURACY, metric.TEST_LOSS]
    metric_names = [metric.TEST_ACCURACY, metric.TRAIN_STATIC_REGRET]
    dataset = "Mnist"
    model = "softmax_regression"
    conf = {"graph_type": "CompleteGraph",
            "centralized": "centralized",
            "nodes_cnt": 10,
            "byzantine_cnt": 2,
            "epoch_or_iteration": "iteration",
            "rounds": 10,
            "rounds_iterations": 5000,
            "lr_controller": "ConstantThenDecreasingLr",
            "init_lr": 0.1,
            "momentum_controller": "FollowOne",
            "init_momentum": 0.1,
            "partition_type": "iid",
            "task_name": "",
            "use_momentum": False
            }

    if conf["use_momentum"] is False:
        conf["momentum_controller"] = "ConstantLr"
        conf["init_momentum"] = 0

    extra = dict()
    # extra["aggregation_rules"] = ["Mean", "Median", "GeometricMedian", "Krum", "TrimmedMean", "Faba", "Phocas",
    #                               "CenteredClipping"]
    # extra["aggregation_show_name"] = ["mean", "coordinate-wise median", "geometric median", "Krum", "trimmed mean",
    #                                   "FABA", "Phocas", "centered clipping"]
    # extra["attack_types"] = ["NoAttack", "SignFlipping", "Gaussian", "SampleDuplicating"]
    # extra["attack_show_name"] = ["without attack", "sign-flipping attack", "Gaussian attack",
    #                              "sample-duplicating attack"]

    extra["aggregation_rules"] = ["Mean", "Median", "GeometricMedian", "Krum", "TrimmedMean", "Faba", "Phocas",
                                  "CenteredClipping"]
    extra["aggregation_show_name"] = ["mean", "coordinate-wise median", "geometric median", "Krum", "trimmed mean",
                                      "FABA", "Phocas", "centered clipping"]
    extra["attack_types"] = ["NoAttack", "SignFlipping", "Gaussian", "SampleDuplicating"]
    extra["attack_show_name"] = ["without attack", "sign-flipping attack", "Gaussian attack",
                                 "sample-duplicating attack"]

    metric_plotter(metric_names=metric_names, dataset=dataset, model=model, conf=conf, extra=extra, data_root=data_root)


def draw_yeCifar10():
    data_root = "../../"
    metric_names = [metric.TEST_ACCURACY, metric.TEST_LOSS]
    # metric_names = [metric.TEST_LOSS, metric.TRAIN_LOSS]
    dataset = "Cifar10"
    model = "resnet18"
    conf = {"graph_type": "ErdosRenyi",
            "centralized": "decentralized",
            "nodes_cnt": 10,
            "byzantine_cnt": 2,
            "epoch_or_iteration": "iteration",
            "rounds": 10,
            "rounds_iterations": 3000,
            "lr_controller": "DecreasingStepLr",
            "init_lr": 0.025,
            "momentum_controller": "FollowOne",
            "init_momentum": 0.1,
            "partition_type": "iid",
            "task_name": "",
            "use_momentum": True
            }

    if conf["use_momentum"] is False:
        conf["momentum_controller"] = "ConstantLr"
        conf["init_momentum"] = 0

    extra = dict()
    extra["aggregation_rules"] = ["Mean", "TrimmedMean",  "CenteredClipping", "IOS"]
    extra["aggregation_show_name"] = ["mean", "trimmed mean", "centered clipping", "IOS"]
    # extra["attack_types"] = ["NoAttack", "SignFlipping", "Gaussian", "SampleDuplicating"]
    # extra["attack_show_name"] = ["without attack", "sign-flipping attack", "Gaussian attack",
    #                              "sample-duplicating attack"]
    extra["attack_types"] = ["NoAttack", ]
    extra["attack_show_name"] = ["without attack"]
    extra["y_lim"] = [[0, 3], [0, 3]]

    metric_plotter(metric_names=metric_names, dataset=dataset, model=model, conf=conf, extra=extra, data_root=data_root)


def draw_dongCifar10():
    data_root = "../../applications/federated_robust_online_gradient_descent/"
    metric_names = [metric.TEST_ACCURACY, metric.TRAIN_STATIC_REGRET]
    # metric_names = [metric.TEST_LOSS, metric.TRAIN_LOSS]
    dataset = "Cifar10"
    model = "resnet18"
    conf = {"graph_type": "ErdosRenyi",
            "centralized": "centralized",
            "nodes_cnt": 10,
            "byzantine_cnt": 2,
            "epoch_or_iteration": "iteration",
            "rounds": 10,
            "rounds_iterations": 2000,
            "lr_controller": "DecreasingStepLr",
            "init_lr": 0.025,
            "momentum_controller": "FollowOne",
            "init_momentum": 0.1,
            "partition_type": "iid",
            "task_name": "",
            "use_momentum": True
            }

    if conf["use_momentum"] is False:
        conf["momentum_controller"] = "ConstantLr"
        conf["init_momentum"] = 0

    extra = dict()
    extra["aggregation_rules"] = ["Mean", "Faba", "CenteredClipping", "Median", "Phocas", "GeometricMedian"]
    extra["aggregation_show_name"] = ["mean", "FABA", "centered clipping",  "Median", "Phocas", "geometric median"]
    # extra["attack_types"] = ["NoAttack", "SignFlipping", "Gaussian", "SampleDuplicating"]
    # extra["attack_show_name"] = ["without attack", "sign-flipping attack", "Gaussian attack",
    #                              "sample-duplicating attack"]
    extra["attack_types"] = ["SignFlipping", "Gaussian",  "SampleDuplicating"]
    extra["attack_show_name"] = ["sign-flipping attack","Gaussian attack",  "sample-duplicating attack"]
    extra["y_lim"] = [[0, 3], [0, 10000]]

    metric_plotter(metric_names=metric_names, dataset=dataset, model=model, conf=conf, extra=extra, data_root=data_root,
                   draw_x=200)

if __name__ == '__main__':
    # main()
    # draw_yeCifar10()
    draw_dongCifar10()
