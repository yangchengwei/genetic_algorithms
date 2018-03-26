
import math
from random import random, randint, uniform

def fitness_function(x1, x2):
    out = 1*( x2 - (5.1/(4*(math.pi**2))*(x1**2)) + (5/math.pi*x1) - (6) )**2 + 10*(1-(1/(8*math.pi)))*math.cos(x1) + 10
#    out = x1+2*x2
#    out = (x1+4)**2+(x2-14)**2
    return out

class Chromosome:
    def __init__(self, gene):
        self.geno = gene
        if gene == 'random':
            self.random_gene()
        self.pheno_x1 = 0
        self.pheno_x2 = 0
        self.fitness = 0
        self.update_fitness()
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

    def update_fitness(self):
        x1_gene = self.geno[::2]
        x2_gene = self.geno[1::2]
        x1_pheno = 0
        x2_pheno = 0
        for i in range(11):
            x1_pheno = x1_pheno*2 + int(x1_gene[i])
            x2_pheno = x2_pheno*2 + int(x2_gene[i])
        x1_pheno = round( x1_pheno/2047*15 + (-5) ,2)
        x2_pheno = round( x2_pheno/2047*15 + (0) ,2)
        self.pheno_x1 = x1_pheno
        self.pheno_x2 = x2_pheno
        self.fitness = fitness_function( x1_pheno, x2_pheno )
        return
        
    def random_gene(self):
        gene = ''
        for i in range(22):
            gene = gene + str(randint(0, 1))
        self.geno = gene
        return
        
class Population:
    def __init__(self, size=2048, crossover_rate=0.7, mutation_rate=0.05):
        self.size = size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.children = []
        for i in range(size):
            self.population.append(Chromosome('random'))
        self.population = sorted(self.population, key=lambda x: x.fitness)
        return
        
    def survive(self):
        ''' Roulette Wheel '''
#        self.population = []
#        sum_fitness = sum([(400-c.fitness) for c in self.children])
#        for i in range(self.size):
#            pick    = uniform(0, sum_fitness)
#            current = 0
#            for survivor in self.children:
#                current = current+(400-survivor.fitness)
#                if current >= pick:
#                    self.population.extend([survivor])
#                    break
#            else:
#                print('ERROR~!')
        ''' Roulette Wheel 2 '''
        self.population = []
        self.children = sorted(self.children, key=lambda x: x.fitness)
        sum_fitness = sum(range(1,len(self.children)+1))
        for i in range(self.size):
            pick    = uniform(0, sum_fitness)
            current = 0
            for survivor in range(len(self.children),0,-1):
                current = current + survivor
                if current >= pick:
                    self.population.extend([self.children[len(self.children)-survivor]])
                    break
            else:
                print('ERROR~!')
        ''' First (size) children survive '''
#        self.population = sorted(self.children, key=lambda x: x.fitness)[:self.size]
        return
        
    def select_parents(self):
        if len(self.population)<2: 
            print('error')
            return None
        return self.population.pop(randint(0,len(self.population)-1)), self.population.pop(randint(0,len(self.population)-1))
        
    def evolve(self):
        self.children = []
        while (len(self.population)>0):
            p1, p2 = self.select_parents()
            self.children.extend([p1, p2])
            if random() <= self.crossover_rate:
                c1, c2 = p1.crossover(p2)
                self.children.extend([c1, c2])
        for idx in range(len(self.children)):
            if random() <= self.mutation_rate:
                self.children.extend([self.children[idx].mutate()])
        self.survive()
        self.population = sorted(self.population, key=lambda x: x.fitness)
        return

if __name__ == "__main__":
    maxGenerations = 50
    times = 50
    every_geno=[]
    every_fitness=[]
    every_x1=[]
    every_x2=[]
    top_time = -1
    top_fitness = 1000
    for time in range(times):
        P = Population(size=2048, crossover_rate=0.7, mutation_rate=0.05)
        for i in range(1, maxGenerations + 1):
            print("Generation %d: %s %f %f %f"%(i, P.population[0].geno,
                                                P.population[0].fitness,
                                                P.population[0].pheno_x1,
                                                P.population[0].pheno_x2))
            P.evolve()

        every_geno.extend([P.population[0].geno])
        every_fitness.extend([P.population[0].fitness])
        every_x1.extend([P.population[0].pheno_x1])
        every_x2.extend([P.population[0].pheno_x2])
        if (P.population[0].fitness < top_fitness):
            top_fitness = P.population[0].fitness
            top_time = time
    for time in range(times):
        print('%s %f %f %f'%(every_geno[time],every_fitness[time],every_x1[time],every_x2[time]))
    print('Final result:')
    print('Genotype : %s'%(every_geno[top_time]))
    print('f(x1,x2) : %f'%(every_fitness[top_time]))
    print('      x1 : %f'%(every_x1[top_time]))
    print('      x2 : %f'%(every_x2[top_time]))
    