import sys
sys.path.insert(0,'../../..')

from utils.plot_utils import *


result_path = "./results/"

runs_count = 3



names = []
names.append("ppo_rnd_1")

for name in names:    
    files = []

    for run in range(runs_count):
        files.append("./models/" + name + "_" + str(run) + "/result/result.log")
    
    plot_rnd(files, result_path, name)



names = []

names.append("ppo_cnd_0")
names.append("ppo_cnd_1")

for name in names:    
    files = []

    for run in range(runs_count):
        files.append("./models/" + name + "_" + str(run) + "/result/result.log")
    
    plot_cnd(files, result_path, name)


labels = []
agents = []
colors = []

labels.append("rnd baseline")
labels.append("cnd mse")
labels.append("cnd vicreg")

agents.append("ppo_rnd_1")
agents.append("ppo_cnd_0")
agents.append("ppo_cnd_1")

colors.append("deepskyblue") 
colors.append("royalblue")
colors.append("blueviolet")

files_runs  = []

for agent in agents:
    runs = []
    for i in range(runs_count):
        runs.append("./models/" + agent + "_" + str(i) + "/result/result.log")
    files_runs.append(runs)


print(files_runs)
plot_summary_score(files_runs, labels, colors, result_path + "summary_score.png")
