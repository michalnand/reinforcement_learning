import RLAgents
import numpy

import matplotlib.pyplot as plt


raw_score_col           = 3
normalized_score_col    = 4

result_path = "./results/"


files = []
files.append("./models/ppo_baseline/result/result.log")
stats_baseline = RLAgents.RLStatsCompute(files) 


files = []
files.append("./models/ppo_rnd/result/result.log")
stats_curiosity = RLAgents.RLStatsCompute(files) 



plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(stats_baseline.mean[0], stats_baseline.mean[raw_score_col], label="baseline with rewards", color='gray')
plt.plot(stats_curiosity.mean[0], stats_curiosity.mean[raw_score_col], label="RND", color='deepskyblue')

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)

