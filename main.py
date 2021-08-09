import mdptoolbox
import numpy as np

low = 0
high = 1
epsilon = pow(10, -5)
rounds = 76
# There are three different fork for the sanme height combination.
states_num = rounds*rounds*3
alpha = 0.45
gamma = 0.5

# generate P.
# four actions: adopt, overrid, wait, match.
P = np.zeros([states_num, states_num, 4])
P[:, 0, 0] = 1
print(P[:, :, 0])
# print(P[:, :, 1])
# print(P[:, :, 2])
# print(P[:, :, 3])
# while high-low > epsilon:
#     rho = (low+high)/2

#     # generate Reward with different rho.
#     R = np.array([[5, 10], [-1, 2]])

#     max_iter = 1000
#     vi = mdptoolbox.mdp.RelativeValueIteration(P, R, max_iter=max_iter)
#     vi.run()
#     print(vi.policy)
