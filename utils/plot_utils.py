import RLAgents
import numpy
from collections import namedtuple


import matplotlib.pyplot as plt
#import networkx as nx
from matplotlib.ticker import MaxNLocator


def _add_plot(axs, labels, colors, stats, idx, from_extended = False):
    for i in range(len(stats)):

        x = stats[i].mean[0]*128/1000000
        if from_extended:
            #x = stats[i].extended_mean[0]*128/1000000
            y = stats[i].extended_mean[idx]
            y_min = stats[i].extended_lower[idx]
            y_max = stats[i].extended_upper[idx]
        else:
            #x = stats[i].mean[0]*128/1000000
            y = stats[i].mean[idx]
            y_min = stats[i].lower[idx]
            y_max = stats[i].upper[idx]

        if numpy.all((y == 0)):
            jitter = 0.06*i
        else:
            jitter = 0  

        axs.plot(x, y + jitter, label=labels[i], linewidth=2.0, color=colors[i], alpha=1.0)
        axs.fill_between(x, y_min, y_max, facecolor=colors[i], alpha=0.5)
        axs.legend()


def plot_summary_score(files_runs, labels, colors, output_file_name, extended_names = ["explored_rooms"], raw_score_only = False):

    stats = []

    for files in files_runs:
        print("processing stats for ", files)
        stat = RLAgents.RLStatsCompute(files, extended_names = extended_names)
        stats.append(stat)

    plt.clf()

    axis_count = 1

    if raw_score_only == False:
        axis_count+= 1
    if len(extended_names) > 0:
        axis_count+= 1

    
 
    if raw_score_only == True:
        fig, axs = plt.subplots(axis_count, 1, figsize=(10, 5))

        _add_plot(axs, labels, colors, stats, 3)

        axs.legend(loc="upper left")
        axs.set_xlabel("samples [milions]", fontweight='bold')
        axs.set_ylabel("score", fontweight='bold')
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.grid(True)
    else:
        fig, axs = plt.subplots(axis_count, 1, figsize=(10, 10))

        _add_plot(axs[0], labels, colors, stats, 3)

        axs[0].legend(loc="upper left")
        axs[0].set_xlabel("samples [milions]", fontweight='bold')
        axs[0].set_ylabel("score", fontweight='bold')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].grid(True)


        _add_plot(axs[1], labels, colors, stats, 4)

        axs[1].legend(loc="upper left")
        axs[1].set_xlabel("samples [milions]", fontweight='bold')
        axs[1].set_ylabel("external reward", fontweight='bold')
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].grid(True)

        if len(extended_names) > 0:
            _add_plot(axs[2], labels, colors, stats, 0, True)

            axs[2].legend(loc="upper left")
            axs[2].set_xlabel("samples [milions]", fontweight='bold')
            axs[2].set_ylabel("explored rooms", fontweight='bold')
            axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[2].grid(True)
        

    fig.tight_layout()
    fig.savefig(output_file_name, dpi = 300, bbox_inches='tight', pad_inches=0.1)


def plot_baseline(files, output_path, output_prefix):
    stats = RLAgents.RLStatsCompute(files)

    plt.clf()

    samples = stats.mean[0]*128/1000000
    
    if "explored_rooms" in stats.extended[0]:
        fig, axs = plt.subplots(3, 1, figsize=(10,10))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10,10))

    axs[0].plot(samples, stats.mean[3], color='red')
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("score", fontweight='bold')
    axs[0].set_xlim(left=0)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)
 

    axs[1].plot(samples, stats.mean[4], color='deepskyblue')
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("external reward", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)


    if "explored_rooms" in stats.extended[0]:
        explored_rooms = []
        for extended in stats.extended[0]:
            explored_rooms.append(int(extended["explored_rooms"]))

        axs[2].plot(samples, explored_rooms, color='limegreen')
        axs[2].set_xlabel("samples [milions]", fontweight='bold')
        axs[2].set_ylabel("explored rooms", fontweight='bold')
        axs[2].set_xlim(left=0)
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].grid(True)


    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_summary.png", dpi = 300)

    


    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10,8))

    axs[0].plot(samples, numpy.clip(stats.mean[6], -0.3, 0.3), color='red')
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("loss actor", fontweight='bold')
    axs[0].set_xlim(left=0)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)

    axs[1].plot(samples, numpy.clip(stats.mean[7], 0, 4), color='limegreen')
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("loss critic", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True) 

    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_all.png", dpi = 300)


    return stats





def plot_rnd(files, output_path, output_prefix, extended_names = ["explored_rooms"]):
    stats = RLAgents.RLStatsCompute(files, extended_names = extended_names)

    plt.clf()

    samples = stats.mean[0]*128/1000000
    
    if "explored_rooms" in stats.extended[0][0]:
        fig, axs = plt.subplots(3, 1, figsize=(10,10))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10,10))

    axs[0].plot(samples, stats.mean[3], color='red')
    axs[0].fill_between(samples, stats.lower[3], stats.upper[3], facecolor='red', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("score", fontweight='bold')
    axs[0].set_xlim(left=0)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)
 

    axs[1].plot(samples, stats.mean[4], color='deepskyblue')
    axs[1].fill_between(samples, stats.lower[4], stats.upper[4], facecolor='deepskyblue', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("external reward", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)

    if "explored_rooms" in stats.extended[0][0]:
        axs[2].plot(samples, stats.extended_mean[0], color='limegreen')
        axs[2].fill_between(samples, stats.extended_lower[0], stats.extended_upper[0], facecolor='limegreen', alpha=0.5)
        axs[2].set_xlabel("samples [milions]", fontweight='bold')
        axs[2].set_ylabel("explored rooms", fontweight='bold')
        axs[2].set_xlim(left=0)
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].grid(True)


    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_summary.png", dpi = 300)

    


    plt.clf()
    fig, axs = plt.subplots(4, 1, figsize=(10,12))

    axs[0].plot(samples, numpy.clip(stats.mean[7], -0.3, 0.3), color='red')
    axs[0].fill_between(samples, numpy.clip(stats.lower[7], -0.3, 0.3), numpy.clip(stats.upper[7], -0.3, 0.3), facecolor='red', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("loss actor", fontweight='bold')
    axs[0].set_xlim(left=0)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)

    axs[1].plot(samples, numpy.clip(stats.mean[8], 0, 4), color='limegreen')
    axs[1].fill_between(samples, numpy.clip(stats.lower[8], 0, 4), numpy.clip(stats.upper[8], 0, 4), facecolor='limegreen', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("loss critic", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True) 

    axs[2].plot(samples, numpy.clip(stats.mean[6], 0, 0.2), color='deepskyblue')
    axs[2].fill_between(samples, numpy.clip(stats.lower[6], 0.0, 0.2), numpy.clip(stats.upper[6], 0.0, 0.2), facecolor='deepskyblue', alpha=0.5)
    axs[2].set_xlabel("samples [milions]", fontweight='bold')
    axs[2].set_ylabel("loss RND", fontweight='bold')
    axs[2].set_xlim(left=0)
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True)

    axs[3].plot(samples, numpy.clip(stats.mean[9], 0.0, 0.2), color='purple')
    axs[3].fill_between(samples, numpy.clip(stats.lower[9], 0.0, 0.2), numpy.clip(stats.upper[9], 0.0, 0.2), facecolor='purple', alpha=0.5)
    axs[3].set_xlabel("samples [milions]", fontweight='bold')
    axs[3].set_ylabel("internal motivation", fontweight='bold')
    axs[3].set_xlim(left=0)
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].grid(True)

  
    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_all.png", dpi = 300)

    return stats










def plot_cnd(files, output_path, output_prefix, extended_names = ["explored_rooms"]):
    stats = RLAgents.RLStatsCompute(files, extended_names = extended_names)

    plt.clf()

    samples = stats.mean[0]*128/1000000
    
    if "explored_rooms" in stats.extended[0][0]:
        fig, axs = plt.subplots(3, 1, figsize=(10,10))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10,10))

    axs[0].plot(samples, stats.mean[3], color='red')
    axs[0].fill_between(samples, stats.lower[3], stats.upper[3], facecolor='red', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("score", fontweight='bold')
    axs[0].set_xlim(left=0)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)
 

    axs[1].plot(samples, stats.mean[4], color='deepskyblue')
    axs[1].fill_between(samples, stats.lower[4], stats.upper[4], facecolor='deepskyblue', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("external reward", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True)


    if "explored_rooms" in stats.extended[0][0]:
        axs[2].plot(samples, stats.extended_mean[0], color='limegreen')
        axs[2].fill_between(samples, stats.extended_lower[0], stats.extended_upper[0], facecolor='limegreen', alpha=0.5)
        axs[2].set_ylabel("explored rooms", fontweight='bold')
        axs[2].set_xlim(left=0)
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].grid(True)


    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_summary.png", dpi = 300)

    


    plt.clf()
    fig, axs = plt.subplots(6, 1, figsize=(10,16))

    axs[0].plot(samples, numpy.clip(stats.mean[9], -0.3, 0.3), color='red')
    axs[0].fill_between(samples, numpy.clip(stats.lower[9], -0.3, 0.3), numpy.clip(stats.upper[9], -0.3, 0.3), facecolor='red', alpha=0.5)
    axs[0].set_xlabel("samples [milions]", fontweight='bold')
    axs[0].set_ylabel("loss actor", fontweight='bold')
    axs[0].set_xlim(left=0)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True)

    axs[1].plot(samples, numpy.clip(stats.mean[10], 0.0, 4.0), color='limegreen')
    axs[1].fill_between(samples, numpy.clip(stats.lower[10], 0.0, 4.0), numpy.clip(stats.upper[10], 0.0, 4.0), facecolor='limegreen', alpha=0.5)
    axs[1].set_xlabel("samples [milions]", fontweight='bold')
    axs[1].set_ylabel("loss critic", fontweight='bold')
    axs[1].set_xlim(left=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True) 

    axs[2].plot(samples, numpy.clip(stats.mean[6], 0.0, 0.2), color='deepskyblue')
    axs[2].fill_between(samples, numpy.clip(stats.lower[6], 0.0, 0.2), numpy.clip(stats.upper[6], 0.0, 0.2), facecolor='deepskyblue', alpha=0.5)
    axs[2].set_xlabel("samples [milions]", fontweight='bold')
    axs[2].set_ylabel("loss CND", fontweight='bold')
    axs[2].set_xlim(left=0)
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True)

    axs[3].plot(samples, numpy.clip(stats.mean[7], 0.0, 0.2), color='deepskyblue')
    axs[3].fill_between(samples, numpy.clip(stats.lower[7], 0.0, 0.2), numpy.clip(stats.upper[7], 0.0, 0.2), facecolor='deepskyblue', alpha=0.5)
    axs[3].set_xlabel("samples [milions]", fontweight='bold')
    axs[3].set_ylabel("loss CND target", fontweight='bold')
    axs[3].set_xlim(left=0)
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].grid(True)

    axs[4].plot(samples, numpy.clip(stats.mean[11], 0.0, 0.2), color='purple')
    axs[4].fill_between(samples, numpy.clip(stats.lower[11], 0.0, 0.2), numpy.clip(stats.upper[11], 0.0, 0.2), facecolor='purple', alpha=0.5)
    axs[4].set_xlabel("samples [milions]", fontweight='bold')
    axs[4].set_ylabel("internal motivation", fontweight='bold')
    axs[4].set_xlim(left=0)
    axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].grid(True)

    axs[5].plot(samples, stats.mean[8], color='purple')
    axs[5].fill_between(samples, stats.lower[8], stats.upper[8], facecolor='purple', alpha=0.5)
    axs[5].set_xlabel("samples [milions]", fontweight='bold')
    axs[5].set_ylabel("target CND magnitude", fontweight='bold')
    axs[5].set_xlim(left=0)
    axs[5].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[5].grid(True)

    fig.tight_layout()
    plt.savefig(output_path + output_prefix + "_all.png", dpi = 300)




    if stats.mean[13].mean() > 0.0:
        extended = True
    else:
        extended = False

    if extended:
        axs[0].plot(samples, stats.mean[13], color='navy')
        axs[0].fill_between(samples, stats.lower[13], stats.upper[13], facecolor='navy', alpha=0.5)
        axs[0].set_xlabel("samples [milions]", fontweight='bold')
        axs[0].set_ylabel("loss symmetry", fontweight='bold')
        axs[0].set_xlim(left=0)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].grid(True)

        axs[1].plot(samples, 100.0*stats.mean[14], color='deepskyblue')
        axs[1].fill_between(samples, stats.lower[14], stats.upper[14], facecolor='deepskyblue', alpha=0.5)
        axs[1].set_xlabel("samples [milions]", fontweight='bold')
        axs[1].set_ylabel("symmetry accuracy [%]", fontweight='bold')
        axs[1].set_xlim(left=0)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].grid(True)

        axs[2].plot(samples, stats.mean[15], color='purple')
        axs[2].fill_between(samples, stats.lower[14], stats.upper[15], facecolor='purple', alpha=0.5)
        axs[2].set_xlabel("samples [milions]", fontweight='bold')
        axs[2].set_ylabel("symmetry magnitude", fontweight='bold')
        axs[2].set_xlim(left=0)
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].grid(True)

        fig.tight_layout()
        plt.savefig(output_path + output_prefix + "_extended.png", dpi = 300)


    return stats
