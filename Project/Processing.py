import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os

x = []
y_pred = [[],[],[]]
y_obs = [[],[],[]]
out = []
counter = 0
flag_new = False
exp_counter = 0
proj_path = os.path.dirname(os.path.abspath('./Processing.py'))

f = open(proj_path + "/Results/Modified_observation/xdpole/nh_18/Inp_adds_g240.txt", "r")
for ind, line in enumerate(f.readlines()):
    if ind % 3 == 0:
        if flag_new:
            flag_new = False
            out.append(pd.DataFrame(np.array([x, y_obs[0], y_obs[1], y_obs[2], y_pred[0], y_pred[1], y_pred[2]]).T,
                                    columns=['step', 'x', 'th_1', 'th_2', 'x_add', 'th_1_add', 'th_2_add']))
            counter = 0
            exp_counter += 1
            y_pred[0].clear()
            y_obs[0].clear()
            y_pred[1].clear()
            y_obs[1].clear()
            y_pred[2].clear()
            y_obs[2].clear()
            x.clear()

        x.append(counter)
        counter += 1
    reg_exp = re.search("(-?\d*\.?\d*e?-?\d*)\ *\t*(-?\d*\.?\d*e?-?\d*)", line)
    obs = float(reg_exp.group(1))
    pred = float(reg_exp.group(2))
    if obs > 1 or obs < -1:
        flag_new = True
    y_pred[ind % 3].append(pred)
    y_obs[ind % 3].append(obs)

print("total number of experiments in procesed file: {}".format(exp_counter+1))

ets = 1
plt.plot(out[ets]["step"], out[ets]["x"],
         out[ets]["step"], (out[ets]["x"]+out[ets]["x_add"]).rolling(2).mean(),
         out[ets]["step"], out[ets]["th_1"],
         out[ets]["step"], (out[ets]["th_1"] + out[ets]["th_1_add"]).rolling(2).mean(),
         out[ets]["step"], out[ets]["th_2"],
         out[ets]["step"], (out[ets]["th_2"] + out[ets]["th_2_add"]).rolling(2).mean())
plt.xlabel("step")
plt.ylabel("value")
plt.legend(['x observer', 'filtered x input',
            'th_1 observer', 'filtered th_1 input',
            'th_2 observer', 'filtered th_2 input'])
plt.show()

