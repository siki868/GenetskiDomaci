import random
import numpy as np
import matplotlib.pyplot as plt
import math



class GA:
    def __init__(self, opseg, pop_vel, max_iter, test_vel, mut_rate):
        self.opseg = opseg
        self.pop_vel = pop_vel
        self.max_iter = max_iter
        self.test_vel = test_vel
        self.mut_rate = mut_rate
    
    def __dekoduj__(self, hromozom, d):
        a =  hromozom[:len(hromozom)//2]
        b =  hromozom[len(hromozom)//2:]
        x = round(-2 + 0.001*int(a, 2), 3)
        y = round(-2 + 0.001*int(b, 2), 3)
        return x, y

    def __inv_random__(self, niz):
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

    def __trosak__(self, hromozom):
        x = hromozom[0]
        y = hromozom[1]
        return (1 + math.pow(x+y+1, 2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))*(30 + math.pow(2*x - 3*y, 2) * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))

    def __mutacija__(self, hromozom, rate):
        if random.random() <= rate:
            #pop[pop.index(hromozom)] = inv_random(hromozom)
            # prvi_deo = ceo[:len(ceo)//2]
            # drugi_deo = ceo[len(ceo)//2:]
            # hromozom[0] = dict_bin[round(-2 + 0.001*int(prvi_deo, 2), 3)]
            # hromozom[1] = dict_bin[round(-2 + 0.001*int(drugi_deo, 2), 3)]
            return self.__inv_random__(hromozom)
        else:
            return hromozom


    def __get_dict__(self, range, inc):
        get_bin = lambda x, n: format(x, 'b').zfill(n)          # f-ja  int -> bin
        float_v_np = np.arange(range[0], range[1]+0.001, inc)   # np niz od -2 do 2 po 0.001 skoku (4k vrednosti)
        _ = np.arange(0, len(float_v_np), 1)                    # obicni brojevi od 0 do 4k koje cemo da pretvorimo u binarne
        binarni = [get_bin(i, 12) for i in _]                   # ^
        f_v = [round(broj, 3) for broj in float_v_np]           # zaokruzivanje na 3 decimale da bi mogli da izvucemo binarnu vrednost preko floata kao kljuc
        
        return dict(zip(f_v, binarni))                          # (-2. : 000000000000) ...


    def __turnir__(self, fja, pop, vel, d):
        z = []
        while len(z) < vel:
            z.append(random.choice(pop))
        najbolji = None
        najbolji_f = None
        for e in z:
            x, y = self.__dekoduj__(e, d)
            ff = fja([x, y])
            if najbolji is None or ff < najbolji_f:
                najbolji_f = ff
                najbolji = e
            return najbolji

    def __ukrsti__(self, h1, h2):
        r = random.randrange(1, len(h1)-1)
        h3 = h1[:r] + h2[r:]
        h4 = h2[:r] + h1[r:]
        return h3, h4
        
    def evaluate(self):
        npop_vel = self.pop_vel
        pop_float = [[round(random.uniform(*self.opseg), 3) for i in range(self.test_vel)] for j in range(self.pop_vel)]

        # Bitan dict
        dict_binarnih = self.__get_dict__(self.opseg, 0.001)

        pop = [dict_binarnih[j[0]] + dict_binarnih[j[1]] for j in pop_float]

        t = 0
        best = None
        best_f = None
        best_ever_f = None
        lista_najboljih = []
        srednje = []
        while best_f != 0 and t < self.max_iter:
            n_pop = pop[:]
            while(len(n_pop) < self.pop_vel+npop_vel) and t < self.max_iter:
                h1 = self.__turnir__(self.__trosak__, pop, 3, dict_binarnih)
                h2 = self.__turnir__(self.__trosak__, pop, 3, dict_binarnih)
                h3, h4 = self.__ukrsti__(h1, h2)
                # print(dekoduj(h3, dict_binarnih))
                h3 = self.__mutacija__(h3, self.mut_rate)
                # print(dekoduj(h3, dict_binarnih), '\n')
                h4 = self.__mutacija__(h4, self.mut_rate)
                n_pop.append(h3)
                n_pop.append(h4)
                pop = sorted(n_pop, key=lambda x : self.__trosak__(self.__dekoduj__(x, dict_binarnih)))[:self.pop_vel]
                f = self.__trosak__(self.__dekoduj__(pop[0], dict_binarnih))
                if best_f is None or best_f > f:
                    best_f = f
                    best = pop[0]
                t += 1
                if best_ever_f is None or best_ever_f > best_f:
                        best_ever_f = best_f
                        best_ever_sol = best
                x, y = self.__dekoduj__(best, dict_binarnih)
                print(f'{t}: ({x}, {y}) loss= {round(best_f, 5)}')
                lista_najboljih.append(best_f)
                sr = [self.__trosak__(self.__dekoduj__(x, dict_binarnih)) for x in n_pop]
                srednje.append(np.mean(sr))
            lista_najboljih = lista_najboljih[:self.max_iter]
        return lista_najboljih, srednje
