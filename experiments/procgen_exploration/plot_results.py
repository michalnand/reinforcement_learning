import sys
sys.path.insert(0,'../..')

from utils.plot_utils import *


result_path = "./results/"

runs_count  = 3

#names = ["climber", "coinrun", "jumper", "caveflyer"]

names = ["climber", "coinrun", "jumper", "caveflyer"]


'''
for name in names:
    
    files = []
    for run in range(1):
        files.append("./" + name + "/models/ppo_baseline_" + str(run) + "/result/result.log")
    plot_baseline(files, result_path, name + "_baseline")
    
    files = []
    for run in range(1):
        files.append("./" + name + "/models/ppo_rnd_" + str(run) + "/result/result.log")
    plot_baseline(files, result_path, name + "_rnd")
    

    files = []
    for run in range(runs_count):
        files.append("./" + name + "/models/ppo_cnd_0_" + str(run) + "/result/result.log")
    plot_cnd(files, result_path, name + "_cnd_0", extended_names = [])

    files = []
    for run in range(runs_count):
        files.append("./" + name + "/models/ppo_cnd_1_" + str(run) + "/result/result.log")
    plot_cnd(files, result_path, name + "_cnd_1", extended_names = [])

    files = []
    for run in range(runs_count):
        files.append("./" + name + "/models/ppo_cnd_2_" + str(run) + "/result/result.log")
    plot_cnd(files, result_path, name + "_cnd_2", extended_names = [])



agents      = ["ppo_baseline", "ppo_rnd", "ppo_cnd_0", "ppo_cnd_1", "ppo_cnd_2"]
labels      = ["baseline", "rnd", "cnd c=0.5", "cnd c=1.0", "cnd c=2.0"]
colors      = ["blue", "green", "orangered", "red", "darkred"]


for name in names:    
    files_runs  = []

    for agent in agents:
        runs = []

        for run in range(runs_count):
            runs.append("./" + name + "/models/" + agent + "_" + str(run) + "/result/result.log")
        files_runs.append(runs)


    plot_summary_score(files_runs, labels, colors, result_path + name + "_score.png", extended_names = [], raw_score_only = True)
'''


agents = ["ppo_cnd_1",  "ppo_cnd_1",    "ppo_cnd_1", "ppo_cnd_1"]
colors = ["blue",       "green",        "red",          "purple"]

files_runs  = []

for i in range(len(names)):   

    name    = names[i] 
    agent   = agents[i] 
    color   = colors[i] 
    
    runs = []
    
    for run in range(runs_count):
        runs.append("./" + name + "/models/" + agent + "_" + str(run) + "/result/result.log")
    files_runs.append(runs)


plot_summary_score(files_runs, names, colors, result_path + "all_score.png", extended_names = [], raw_score_only = True)


print("done")

