import numpy as np
import stim

from games.hardstabilizer2_0 import StabilizerEnv

for n in range(2, 15):
    num_moves_list = []
    for _ in range(500):
        env = StabilizerEnv(n); env.reset()
        num_moves = 0
        while env.step(np.random.choice([0, 1]))[2] is False:
            num_moves += 1
        num_moves_list.append(num_moves)

    print(n, np.mean(num_moves_list))
