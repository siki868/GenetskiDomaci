# Prvi domaci iz genetskih algoritama (Minimizacija funkcija) - Nikola Dimitrijevic RN-47/16
# B - Binarni genetski algoritam                                (Vrtsa algoritma)
# J - Jednotackasto ukrstanje na proizvoljnoj tacki ukrstanja   (Vrsta ukrstanja)
# I - Inverzija                                                 (Vrsta mutacije)
# G - Goldstein-Price Function                                  (Vrsta funkcije koja treba da se minimizuje)


# Inverzija podsetnik [1,2,3,4,5,6]  ->  [1,5,4,3,2,6] 
#                        ^     ^


import numpy as np
import matplotlib.pyplot as plt
import math
import random

mut_rates = 0.1


def trosak(hromozom):
    x = hromozom[0]
    y = hromozom[1]
    return (1 + math.pow(x+y+1, 2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))*(30 + math.pow(2*x - 3*y, 2) * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))

def mutacija(hromozom, rate):
    if random.random() <= rate:
        hromozom[0], hromozom[1] = hromozom[1], hromozom[0]



if __name__ == '__main__':
    # goldstein_v = np.vectorize(trosak)
    
    # x = np.arange(-2, 2, 0.002)
    # y = np.arange(-2, 2, 0.002)
    # X, Y = np.meshgrid(x, y)
    # Z = goldstein_v(X, Y)
    
    # plt.figure(num=1, figsize=(10,7))
    # plt.clf()
    # ctr = plt.contour(X, Y, Z, levels=[1e-1,5e-1,1e0,5e0,1e1,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,1e2,11e1])
    # plt.axes().set_aspect('equal')
    # plt.colorbar(ctr)
    # plt.title('Goldestein funkcija')
    # plt.show()



    opseg = [-2, 2]
    test_vel = 2
    pop_vel = 100
    pop = [[random.uniform(*opseg) for i in range(test_vel)] for j in range(pop_vel)]

