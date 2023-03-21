from mpl_toolkits.axisartist.axislines import AxesZero
import matplotlib.pyplot as plt
import numpy as np

# Function drawing Gershgorin circles
def Gershgorin(A, rows=True):
    col, style = 'blue', 'dashed'
    if(not rows): 
        A = np.transpose(A)
        col, style = 'red', 'dotted'
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if(i == j): c = A[i][j]
            r = np.absolute(np.sum(A[i]) - c)
        ax.plot(c + r * np.cos(an), r * np.sin(an), color=col, linestyle = style)
        plt.scatter(c, 0, color=col)

A1 = np.array([[-2, -1, 0], [2, 0 , 0], [0, 0, 2]])
A2 = np.array([[5, 1, 1], [0, 6 , 1], [0, 0, -5]])
A3 = np.array([[5.2, 0.6, 2.2], [0.6, 6.4 , 0.5], [2.2, 0.5, 4.7]])

fig = plt.figure()
ax = fig.add_subplot(axes_class=AxesZero)

for direction in ["xzero", "yzero"]:

    ax.axis[direction].set_visible(True)
    
for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)


an = np.linspace(0, 2 * np.pi, 100)
ax.set_aspect('equal', 'box')
plt.grid(True)

Gershgorin(A1)
Gershgorin(A1, False)

for i in np.linalg.eig(A1)[0]:
    plt.scatter(i.real, i.imag, color = 'green')
plt.show()

ax.set_aspect('equal')
ax.set(xlim=(-8, 8), ylim=(-3, 3))
plt.grid(True)

Gershgorin(A2)
Gershgorin(A2, False)

for i in np.linalg.eig(A2)[0]:
    plt.scatter(i.real, i.imag, color = 'green')
    
plt.show()

ax.set_aspect('equal', 'box')
plt.grid(True)

Gershgorin(A3)
Gershgorin(A3, False)

for i in np.linalg.eig(A3)[0]:
    plt.scatter(i.real, i.imag, color = 'green')
    
plt.show()