import mdptoolbox
import mdptoolbox.example
from scipy.sparse import csr_matrix as sparse

P, R = mdptoolbox.example.forest(S=2400, r1=4, r2=2)
rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)

P = [sparse(P[0]), sparse(P[1])]
R = sparse(R)
# print(P)
print(R)
# print(R.shape)
rvi.run()
rvi.average_reward
