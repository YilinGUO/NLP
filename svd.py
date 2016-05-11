import numpy as np
import matplotlib.pyplot as plt
A = [[1, 1, 1, 0, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1]]
U,s,V = np.linalg.svd(A, full_matrices=False) # SVD decomposition of A
U = -U[:,[0,1]];
V = -V[[0,1], :]
s= np.diag([s[0], s[1]])
cols =  np.dot(np.dot(map(list, zip(*A)), U), s)
rows =  np.dot(np.dot(A, V.T), s)
fig, ax = plt.subplots()
x = np.concatenate([cols[:, 0], rows[:, 0]])
y = np.concatenate([cols[:, 1], rows[:, 1]])
ax.scatter(x, y)
n = ['T1', 'T2', 'T3', 'T4', 'T5', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
for i, txt in enumerate(n):
	ax.annotate(txt, (x[i],y[i]))
plt.show()
