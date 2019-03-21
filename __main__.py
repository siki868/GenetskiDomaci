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



def dekoduj(hromozom, d):
    a =  hromozom[:len(hromozom)//2]
    b =  hromozom[len(hromozom)//2:]
    x = round(-2 + 0.001*int(a, 2), 3)
    y = round(-2 + 0.001*int(b, 2), 3)
    return x, y

def inv_random(niz):
    a = random.randint(0, len(niz)+1) 
    b = random.randint(0, len(niz)+1) 
    while a == b or math.fabs(a-b) == 1 : 
        a = random.randint(0, len(niz)+1) 
        b = random.randint(0, len(niz)+1) 
    if a > b: 
        a, b = b, a
    s = list(niz) 
    s[a:b] = s[a:b][::-1]
    return ''.join(s)

def trosak(hromozom):
    x = hromozom[0]
    y = hromozom[1]
    return (1 + math.pow(x+y+1, 2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))*(30 + math.pow(2*x - 3*y, 2) * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))

def mutacija(hromozom, rate):
    if random.random() <= rate:
        #pop[pop.index(hromozom)] = inv_random(hromozom)
        # prvi_deo = ceo[:len(ceo)//2]
        # drugi_deo = ceo[len(ceo)//2:]
        # hromozom[0] = dict_bin[round(-2 + 0.001*int(prvi_deo, 2), 3)]
        # hromozom[1] = dict_bin[round(-2 + 0.001*int(drugi_deo, 2), 3)]
        return inv_random(hromozom)
    else:
        return hromozom


def get_dict(range, inc):
    get_bin = lambda x, n: format(x, 'b').zfill(n)  # f-ja  int -> bin
    float_v_np = np.arange(range[0], range[1]+0.001, inc) # np niz od -2 do 2 po 0.001 skoku (4k vrednosti)
    _ = np.arange(0, len(float_v_np), 1)            # obicni brojevi od 0 do 4k koje cemo da pretvorimo u binarne
    binarni = [get_bin(i, 12) for i in _]           # ^
    f_v = [round(broj, 3) for broj in float_v_np]   # zaokruzivanje na 3 decimale da bi mogli da izvucemo binarnu vrednost preko floata kao kljuc
    
    return dict(zip(f_v, binarni))                  # (-2. : 000000000000) ...


def turnir(fja, pop, vel, d):
    z = []
    while len(z) < vel:
        z.append(random.choice(pop))
    najbolji = None
    najbolji_f = None
    for e in z:
        x, y = dekoduj(e, d)
        ff = fja([x, y])
        if najbolji is None or ff < najbolji_f:
               najbolji_f = ff
               najbolji = e
        return najbolji

def ukrsti(h1, h2):
    r = random.randrange(1, len(h1)-1)
    h3 = h1[:r] + h2[r:]
    h4 = h2[:r] + h1[r:]
    return h3, h4

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
    npop_vel = 100
    best = None
    best_f = None
    best_ever_f = None
    t = 0
    max_iter = 5000
    # zaokruzivanje na 3 decimale jer hocu da ih spojim u jedan broj koji ce da postane binaran
    # 3 decimale za vrednosti od -2  do 2 je 4000 brojeva sto znaci da nam treba 12 digit-a 
    pop = [[round(random.uniform(*opseg), 3) for i in range(test_vel)] for j in range(pop_vel)]

    # bbitan dict
    dict_binarnih = get_dict(opseg, 0.001)

    pop = [dict_binarnih[j[0]] + dict_binarnih[j[1]] for j in pop]

    # testiranje mutacija
    # print('Spreman!\n')
    # for hromozom in pop:
    #     ind = pop.index(hromozom)
    #     # test_d = dekoduj(hromozom, dict_binarnih)
    #     print(hromozom)
    #     mutacija(hromozom, mut_rates, dict_binarnih, pop)
    #     print(pop[ind])
    #     if(input() == 'd'):
    #         continue
    #     else:
    #         break


    while best_f != 0 and t < max_iter:
        n_pop = pop[:]
        while(len(n_pop) < pop_vel+npop_vel):
            h1 = turnir(trosak, pop, 3, dict_binarnih)
            h2 = turnir(trosak, pop, 3, dict_binarnih)
            h3, h4 = ukrsti(h1, h2)
            h3 = mutacija(h3, mut_rates)
            h4 = mutacija(h4, mut_rates)
            n_pop.append(h3)
            n_pop.append(h4)
            pop = sorted(n_pop, key=lambda x : trosak(dekoduj(x, dict_binarnih)))[:pop_vel]
            f = trosak(dekoduj(pop[0], dict_binarnih))
            if best_f is None or best_f > f:
                best_f = f
                best = pop[0]
            t += 1
            if best_ever_f is None or best_ever_f > best_f:
                    best_ever_f = best_f
                    best_ever_sol = best
            x, y = dekoduj(best, dict_binarnih)
            print(f'{t}: ({x}, {y}) loss= {round(best_f, 5)}')