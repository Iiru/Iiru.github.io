import numpy as np
from numpy.linalg import inv


A = np.random.randint(1,10, size=(2,2))
B = np.random.randint(1,10, size=(2,2))


C = np.matrix(A)
D = np.matrix(B)

print(C)
print(D)



E = C.transpose()
for row in E:
  print(row)

F = D.transpose()
for row in F:
  print(row)

Cinv = inv(C)

for row in Cinv:
  print(row)
  
Dinv = inv(D)

for row in Dinv:
  print(row)


import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt

a = np.arange(0.0, 10.0, 0.01)
b = np.sin(2*np.pi*a)
c = np.cos(2*np.pi*a)


plt.plot(a,b)
plt.plot(a,c)

plt.show()
plt.cla()



import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, rand


a = np.random.randint(1, 100, size=1000)


plt.hist(a, bins=100)
plt.show()
plt.cla()



import matplotlib.pyplot as plt
from numpy.random import normal, rand

x = normal(size=1000)
y = normal(size=1000)

plt.scatter(x,y)
plt.show()
