import sys
sys.path.insert(0,'../../..')

from utils.plot_utils import *


result_path = "./results/"

runs_count = 3





labels = []
agents = []
colors = []

labels.append("noise")
labels.append("tile mask + noise")
labels.append("random conv + tile mask + noise")

agents.append("ppo_cnd_100")
agents.append("ppo_cnd_101")
agents.append("ppo_cnd_102")

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
plot_summary_score(files_runs, labels, colors, result_path + "cnd_vicreg_augmentations.png")




labels = []
agents = []
colors = []

labels.append("symmetric, ELU,  0 hidden")
labels.append("symmetric, ELU,  1 hidden")
labels.append("asymetric, ELU,  0 hidden, 3 hidden")
labels.append("symmetric, ReLU, 1 hidden")

agents.append("ppo_cnd_101")
agents.append("ppo_cnd_103")
agents.append("ppo_cnd_104")
agents.append("ppo_cnd_105")

colors.append("deepskyblue") 
colors.append("royalblue")
colors.append("blueviolet")
colors.append("purple")

files_runs  = []

for agent in agents:
    runs = []
    for i in range(runs_count):
        runs.append("./models/" + agent + "_" + str(i) + "/result/result.log")
    files_runs.append(runs)


print(files_runs)
plot_summary_score(files_runs, labels, colors, result_path + "cnd_vicreg_architecture.png")






labels = []
agents = []
colors = []

labels.append("c=0.25")
labels.append("c=0.5")
labels.append("c=1.0")

agents.append("ppo_cnd_106")
agents.append("ppo_cnd_107")
agents.append("ppo_cnd_108")

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
plot_summary_score(files_runs, labels, colors, result_path + "cnd_vicreg_scaling.png")
