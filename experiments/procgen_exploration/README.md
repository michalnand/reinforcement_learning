# procgen - solving hard seeds

there are hard exploration seeds in original procgen

source [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/pdf/1912.01588.pdf)

see. section **B.1. Evaluating Exploration** for more details

this seeds are **unable to solve** with baseline PPO (or RND) in 200M samples, but CND
solve them in just 64M samples

# 4 tested procgen envs

## caveflyer climber coinrun jumper
<img src="doc/caveflyer.png" width="200">
<img src="doc/climber.png" width="200">
<img src="doc/coinrun.png" width="200">
<img src="doc/jumper.png" width="200">

 
# results

## caveflyer

![animation](doc/caveflyer.gif)

![graph](results/caveflyer_score.png)
 

## climber

![animation](doc/climber.gif)

![graph](results/climber_score.png)


## coinrun

![animation](doc/coinrun.gif)

![graph](results/coinrun_score.png)


## jumper

![animation](doc/jumper.gif)

![graph](results/jumper_score.png)






