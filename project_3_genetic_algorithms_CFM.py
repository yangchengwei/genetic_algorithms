
import math
import time
import matplotlib.pyplot as plt
from random import random, randint, uniform

def function_f1(x1, x2):
    out = x2+10**(-5)*(x2-x1)**2-1
    return out

def function_f2(x1, x2):
    out = 1/(27*math.sqrt(3))*((x1-3)**2-9)*x2**3
    return out

def function_f3(x1, x2):
    out = (1/3)*(x1-2)**3+x2-11/3
    return out

class Individual:
    def __init__(self, x1=0.0, x2=0.0, random_init=False):
        if random_init:
            self.x1 = uniform(0,6)
            self.x2 = uniform(0,math.sqrt(3))
            while(self.x2<0 or abs(self.x1-3)>(1-self.x2/math.sqrt(3))*3):
                self.x1 = uniform(0,6)
                self.x2 = uniform(0,math.sqrt(3))
        else:
            self.x1 = x1
            self.x2 = x2
            
        self.value = 0.0
        self.fitness = 0
        self.update_value()
        return
        
    def crossover(self, mate):
        ''' Arithmetical crossover '''
        a = uniform(0,1)
        new1_x1 = self.x1*a + mate.x1*(1-a)
        new1_x2 = self.x2*a + mate.x2*(1-a)
        new2_x1 = self.x1*(1-a) + mate.x1*a
        new2_x2 = self.x2*(1-a) + mate.x2*a
        return Individual(new1_x1,new1_x2), Individual(new2_x1,new2_x2)
        
    def mutate(self):
        ''' Uniform mutation x1,x2 '''
        if randint(0,1):
            W = (1-self.x2/math.sqrt(3))*3
            new_x1 = uniform(3-W,3+W)
            new_x2 = self.x2
        else:
            H = math.sqrt(3)-abs(self.x1/3-1)*math.sqrt(3)
            new_x1 = self.x1
            new_x2 = uniform(0,H)
        ''' Boundary mutation x1,x2 '''
#        if randint(0,1):
#            W = (1-self.x2/math.sqrt(3))*3
#            new_x1 = 3-W+randint(0,1)*2*W
#            new_x2 = self.x2
#        else:
#            H = math.sqrt(3)-abs(self.x1/3-1)*math.sqrt(3)
#            new_x1 = self.x1
#            new_x2 = randint(0,1)*H
        ''' Uniform mutation v1,v2 '''
#        if randint(0,1):
#            new_rate1 = uniform(0,1-self.rate2)
#            new_rate2 = self.rate2
#            new_x1 = new_rate1*6 + self.rate2*3
#            new_x2 = self.x2
#        else:
#            new_rate1 = self.rate1
#            new_rate2 = uniform(0,1-self.rate1)
#            new_x1 = new_rate1*6 + new_rate2*3
#            new_x2 = new_rate2*(math.sqrt(3))
        ''' Boundary mutation v1,v2 '''
#        if randint(0,1):
#            new_rate1 = randint(0,1)*(1-self.rate2)
#            new_rate2 = self.rate2
#            new_x1 = new_rate1*6 + self.rate2*3
#            new_x2 = self.x2
#        else:
#            new_rate1 = self.rate1
#            new_rate2 = randint(0,1)*(1-self.rate1)
#            new_x1 = new_rate1*6 + new_rate2*3
#            new_x2 = new_rate2*(math.sqrt(3))
        return Individual(new_x1,new_x2)
        
    def update_value(self):
        self.rate2 = self.x2/(math.sqrt(3))
        self.rate1 = (self.x1-(math.sqrt(3))*self.x2)/6.0
        if   0<=self.x1 and self.x1<2:
            self.value = function_f1(self.x1,self.x2)
        elif 2<=self.x1 and self.x1<4:
            self.value = function_f2(self.x1,self.x2)
        elif 4<=self.x1 and self.x1<=6:
            self.value = function_f3(self.x1,self.x2)
        else:
            print('update_value error !')
        return
        
class Population:
    def __init__(self, size=128, crossover_rate=0.7, mutation_rate=0.05):
        self.size = size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.children = []
        for i in range(size):
            self.population.append(Individual(random_init=True))
        self.population = sorted(self.population, key=lambda x: x.value)
        return
        
    def survive(self):
        ''' Roulette Wheel '''
#        self.population = []
#        sum_value = sum([(400-c.value) for c in self.children])
#        for i in range(self.size):
#            pick    = uniform(0, sum_value)
#            current = 0
#            for survivor in self.children:
#                current = current+(400-survivor.value)
#                if current >= pick:
#                    self.population.extend([survivor])
#                    break
#            else:
#                print('ERROR~!')
        ''' Roulette Wheel 2 '''
#        self.population = []
#        self.children = sorted(self.children, key=lambda x: x.value)
#        self.population.extend([self.children[0]])
#        self.children.remove(self.children[0])
#        L = len(self.children)
#        for i in range(L):
#            self.children[i].fitness = L - i
#        sum_fitness = sum(range(1,L+1))
#        for i in range(self.size-1):
#            pick    = uniform(0, sum_fitness)
#            current = 0
#            for survivor in self.children:
#                current = current + survivor.fitness
#                if current >= pick:
#                    self.population.extend([survivor])
#                    self.children.remove(survivor)
#                    sum_fitness = sum_fitness - survivor.fitness
#                    break
#            else:
#                print('survive error !')
#                print('i= %d pick= %f current= %f'%(i, pick, current))
        ''' First (size) children survive '''
#        self.population = sorted(self.children, key=lambda x: x.value)[:self.size]
        return
        
    def select_parents(self):
        if len(self.population)<2: 
            print('select_parents error !')
            return None
        return self.population.pop(randint(0,len(self.population)-1)), self.population.pop(randint(0,len(self.population)-1))
    
    def children_replace(self, child):
        ''' Crowding factor model '''
        min_distance = 6
        min_idx = -1
        L = len(self.children)
        for i in range(L):
            distance = abs(self.children[i].x1-child.x1) \
                     + abs(self.children[i].x2-child.x2)
            if distance<min_distance:
                min_distance = distance
                min_idx = i
        if self.children[min_idx].value>child.value:
            self.children[min_idx]=child
        return
    
    def evolve(self):
        self.children = self.population.copy()
        while (len(self.population)>0):
            p1, p2 = self.select_parents()
            if random() <= self.crossover_rate:
                c1, c2 = p1.crossover(p2)
                self.children_replace(c1)
                self.children_replace(c2)
        for idx in range(len(self.children)):
            if random() <= self.mutation_rate:
                self.children_replace(self.children[idx].mutate())
#        self.survive()
        self.population = self.children.copy()
        self.population = sorted(self.population, key=lambda x: x.value)
        return
        
    def plot(self, generation=0):
        for i in range(self.size):
            plt.plot(self.population[i].x1, self.population[i].x2,'bo', markersize=2)
        plt.xlim(-1,7)
        plt.ylim(-2,4)
        plt.savefig('64_Bnd_generation%03d.png'%(generation))
        plt.show()
        return

if __name__ == "__main__":
    maxGenerations = 100
    times = 100
    every_value=[]
    every_x1=[]
    every_x2=[]
    set_minimum = []
    top_idx = -1
    top_value = 1000
    correct = 0
    min_times = 0
    Tstart = time.time()
    for k in range(times):
        P = Population(size=64, crossover_rate=0.7, mutation_rate=0.1)
        for i in range(1, maxGenerations + 1):
#            print("Generation %d: %f %f %f"%(i, P.population[0].x1,
#                                                P.population[0].x2,
#                                                P.population[0].value))
#            set_minimum.append(P.population[0].value)
            ''' plot individual distribution '''
#            P.plot(generation=i)
            P.evolve()
        ''' compute the minimum found times '''
        for individual in P.population:
            if (individual.x1-0)**2+(individual.x2-0)**2<0.0001:
                min_times = min_times +1
                break
        for individual in P.population:
            if (individual.x1-3)**2+(individual.x2-math.sqrt(3))**2<0.0001:
                min_times = min_times +1
                break
        for individual in P.population:
            if (individual.x1-4)**2+(individual.x2-0)**2<0.0001:
                min_times = min_times +1
                break
    Tend = time.time()
    ''' plot convergence figure '''
#    for i in range(times):
#        set_minimum[i]=sum(set_minimum[i::maxGenerations])/times
#    plt.plot(list(range(1,maxGenerations+1)),set_minimum[:maxGenerations])
#    plt.ylim(-1,-0.9)
#    plt.savefig('./CFM_U_%d'%(P.size))
    print('Find %d global minimum in total %d global minimum'%(min_times, times*3))
    print('Time : %f'%(Tend-Tstart))