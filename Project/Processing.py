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
st_count = 0
proj_path = os.path.dirname(os.path.abspath('./Processing.py'))

f = open(proj_path + "/results/Modified_observation/variation_2/xdpole/Inp_adds_g18.txt", "r")
for ind, line in enumerate(f.readlines()):
    if ind % 3 == 0:
        if flag_new and len(x) > 2:
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
    reg_exp = re.search("0\t*\ *(-?\d*\.?\d*e?-?\d*)\ *\t*(-?\d*\.?\d*e?-?\d*)", line)
    obs = float(reg_exp.group(1))
    pred = float(reg_exp.group(2))
    if pred == 0:
        st_count += 1
    if st_count == 4:
        st_count = 1
        flag_new = True
    # if (ind % 3 == 0) and (obs > 2.4 or obs < -2.4 or counter >= 1000):
    #     flag_new = True
    # if (ind % 3 != 0) and (obs > np.pi/5 or obs < -np.pi/5 or counter >= 1000):
    #     flag_new = True
    y_pred[ind % 3].append(pred)
    y_obs[ind % 3].append(obs)

print("total number of experiments in procesed file: {}".format(exp_counter+1))

# for i in range(len(out)):

length_of_exp = [len(out[x]) for x in range(len(out))]
ets = length_of_exp.index(max(length_of_exp))

plt.plot(out[ets]["step"], out[ets]["x"],
         out[ets]["step"][0:], (out[ets]["x"][0:-1] + out[ets]["x_add"][1:]),
         out[ets]["step"], out[ets]["th_1"],
         out[ets]["step"][0:], (out[ets]["th_1"][0:-1] + out[ets]["th_1_add"][1:]),
         out[ets]["step"], out[ets]["th_2"],
         out[ets]["step"][0:], (out[ets]["th_2"][0:-1] + out[ets]["th_2_add"][1:]))
plt.xlabel("step")
plt.ylabel("value")
plt.legend(['x observer', 'x predicted',
            'th_1 observer', 'th_1 predicted',
            'th_2 observer', 'th_2 predicted'])
plt.ylim([-2.5, 2.5])
plt.show()
print("Shown exp with index: {}".format(ets))
# plt.plot(x, y_obs[0],
#          x, y_pred[0],
#          x, y_obs[1],
#          x, y_pred[1],
#          x, y_obs[2],
#          x, y_pred[2])
# plt.legend(range(6))
#
# plt.show()