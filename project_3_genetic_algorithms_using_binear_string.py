
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

class Chromosome:
    def __init__(self, gene=''):
        self.pheno_x1 = 0
        self.pheno_x2 = 0
        self.value = 0
        self.fitness = 0
        self.feasible = True
        if gene == 'random':
            self.random_gene()
            self.update_value()
            while self.feasible==False:
                self.random_gene()
                self.update_value()
        else:
            self.geno = gene
            self.update_value()
        return
    
    def crossover(self, mate):
        pivot = randint(0, len(self.geno) - 1)
        gene1 = self.geno[:pivot] + mate.geno[pivot:]
        gene2 = mate.geno[:pivot] + self.geno[pivot:]
        return Chromosome(gene1), Chromosome(gene2)
    
    def mutate(self):
        gene = self.geno
        idx = randint(0, len(gene) - 1)
        gene = gene[:idx] + str((int(gene[idx])+1)%2) + gene[idx+1:]
        return Chromosome(gene)

    def update_value(self):
        x1_gene = self.geno[0:10]
        x2_gene = self.geno[10:18]
        x1_pheno = 0
        x2_pheno = 0
        ''' Binary code'''
#        for i in range(11):
#            x1_pheno = x1_pheno*2 + int(x1_gene[i])
#            x2_pheno = x2_pheno*2 + int(x2_gene[i])
        ''' Gray code '''
        x1_flip = False
        x2_flip = False
        for i in range(10):
            x1_read = int(x1_gene[i]) if not x1_flip else (int(x1_gene[i])+1)%2
            x1_pheno = x1_pheno*2 + x1_read
            x1_flip = True if x1_read==1 else False
        for i in range(8):
            x2_read = int(x2_gene[i]) if not x2_flip else (int(x2_gene[i])+1)%2
            x2_pheno = x2_pheno*2 + x2_read
            x2_flip = True if x2_read==1 else False
        
        x1_pheno = x1_pheno/1024*6
        x2_pheno = x2_pheno/256*math.sqrt(3)
        self.pheno_x1 = x1_pheno
        self.pheno_x2 = x2_pheno
        if x2_pheno<0 or abs(x1_pheno-3)>(1-x2_pheno/math.sqrt(3))*3:
            self.feasible = False
            return
        else:
            self.feasible = True
        if   0<=x1_pheno and x1_pheno<2:
            self.value = function_f1(x1_pheno,x2_pheno)
        elif 2<=x1_pheno and x1_pheno<4:
            self.value = function_f2(x1_pheno,x2_pheno)
        elif 4<=x1_pheno and x1_pheno<=6:
            self.value = function_f3(x1_pheno,x2_pheno)
        else:
            print('update_value error !')
        return
        
    def random_gene(self):
        gene = ''
        for i in range(18):
            gene = gene + str(randint(0, 1))
        self.geno = gene
        return
        
class Population:
    def __init__(self, size=64, crossover_rate=0.7, mutation_rate=0.1):
        self.size = size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.children = []
        for i in range(size):
            self.population.append(Chromosome('random'))
        self.population = sorted(self.population, key=lambda x: x.value)
        return
        
    def survive(self):
        ''' Roulette Wheel '''
#        self.population = []
#        sum_fitness = sum([(400-c.value) for c in self.children])
#        for i in range(self.size):
#            pick    = uniform(0, sum_fitness)
#            current = 0
#            for survivor in self.children:
#                current = current+(400-survivor.value)
#                if current >= pick:
#                    self.population.extend([survivor])
#                    break
#            else:
#                print('ERROR~!')
        ''' Roulette Wheel 2 '''
        self.population = []
        self.children = sorted(self.children, key=lambda x: x.value)
        self.population.extend([self.children[0]])
        self.children.remove(self.children[0])
        L = len(self.children)
        for i in range(L):
            self.children[i].fitness = L - i
        sum_fitness = sum(range(1,L+1))
        for i in range(self.size-1):
            pick    = uniform(0, sum_fitness)
            current = 0
            for survivor in self.children:
                current = current + survivor.fitness
                if current >= pick:
                    self.population.extend([survivor])
                    self.children.remove(survivor)
                    sum_fitness = sum_fitness - survivor.fitness
                    break
            else:
                print('survive error !')
                print('i= %d pick= %f current= %f'%(i, pick, current))
        ''' First (size) children survive '''
#        self.population = sorted(self.children, key=lambda x: x.value)[:self.size]
        return
        
    def select_parents(self):
        if len(self.population)<2: 
            print('select_parents error !')
            return None
        return self.population.pop(randint(0,len(self.population)-1)), self.population.pop(randint(0,len(self.population)-1))
        
    def evolve(self):
        self.children = []
        while (len(self.population)>0):
            p1, p2 = self.select_parents()
            self.children.extend([p1, p2])
            if random() <= self.crossover_rate:
                c1, c2 = p1.crossover(p2)
                if c1.feasible:
                    self.children.extend([c1])
                if c2.feasible:
                    self.children.extend([c2])
        for idx in range(len(self.children)):
            if random() <= self.mutation_rate:
                cm = self.children[idx].mutate()
                if cm.feasible:
                    self.children.extend([cm])
        self.survive()
        self.population = sorted(self.population, key=lambda x: x.value)
        return
        
    def plot(self,generation=0):
        for i in range(self.size):
            plt.plot(self.population[i].pheno_x1, self.population[i].pheno_x2,'bo', markersize=2)
        plt.xlim(-1,7)
        plt.ylim(-2,4)
        plt.savefig('generation%03d.png'%(generation))
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
#            print("Generation %d: %f %f %f"%(i, P.population[0].pheno_x1,
#                                                P.population[0].pheno_x2,
#                                                P.population[0].value))
#            set_minimum.append(P.population[0].value)
            ''' plot individual distribution '''
#            P.plot(generation=i)
            P.evolve()
        ''' compute the minimum found times '''
        for individual in P.population:
            if (individual.pheno_x1-0)**2+(individual.pheno_x2-0)**2<0.0001:
                min_times = min_times +1
                break
        for individual in P.population:
            if (individual.pheno_x1-3)**2+(individual.pheno_x2-math.sqrt(3))**2<0.0001:
                min_times = min_times +1
                break
        for individual in P.population:
            if (individual.pheno_x1-4)**2+(individual.pheno_x2-0)**2<0.0001:
                min_times = min_times +1
                break
    Tend = time.time()
    ''' plot convergence figure '''
#    for i in range(times):
#        set_minimum[i]=sum(set_minimum[i::maxGenerations])/times
#    plt.plot(list(range(1,maxGenerations+1)),set_minimum[:maxGenerations])
#    plt.ylim(-1,-0.9)
#    plt.savefig('./GABinary_%d'%(P.size))
    print('Find %d global minimum in total %d global minimum'%(min_times, times*3))
    print('Time : %f'%(Tend-Tstart))
    