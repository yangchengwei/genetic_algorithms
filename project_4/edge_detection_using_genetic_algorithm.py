
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint, uniform

CHANGE = False

#We, Wd, Wc, Wf, Wt = 1, 2, 0.5, 3, 6.51
We, Wd, Wc, Wf,  = 0.5, 2, 0.1, 0
Wt = 2*Wf - Wc + Wd - We + 1
TARGET = cv2.imread("./test_image/test_image2.jpg", 0)/255
N = np.shape(TARGET)[0]    # image size = N * N
DISSIM = TARGET*(-2.0)
DISSIM[1:N//2,:] = DISSIM[1:N//2,:] + TARGET[0:N//2-1,:]
DISSIM[:,1:N//2] = DISSIM[:,1:N//2] + TARGET[:,0:N//2-1]
DISSIM[N//2:N-1,:] = DISSIM[N//2:N-1,:] + TARGET[N//2+1:N,:]
DISSIM[:,N//2:N-1] = DISSIM[:,N//2:N-1] + TARGET[:,N//2+1:N]
DISSIM[0,:] = 0
DISSIM[:,0] = 0
DISSIM[N-1,:] = 0
DISSIM[:,N-1] = 0
DISSIM = np.abs(DISSIM)
DISSIM[DISSIM>1]=1
cv2.imshow("TARGET", TARGET)
cv2.imshow("DISSIM", DISSIM)
print(TARGET.max())
print(DISSIM.max())
print(TARGET.min())
print(DISSIM.min())
#cv2.imwrite('b.jpg',DISSIM*255)
#cv2.waitKey(0)

class Individual:
    def __init__(self, edge_map=None, random_init=False,
                 subregion=(0,0), grid_size=0):
        self.N = grid_size
        self.subregion = subregion
        self.sX, self.eX = subregion[0]*grid_size, (subregion[0]+1)*grid_size
        self.sY, self.eY = subregion[1]*grid_size, (subregion[1]+1)*grid_size
        self.target = TARGET[self.sX:self.eX,self.sY:self.eY]
        self.dissim = DISSIM[self.sX:self.eX,self.sY:self.eY]
        self.value = 0.0
        self.fitness = 0
        if random_init:
            self.edgeMap = np.random.randint( 0, 2, [self.N, self.N] )
        else:
            self.edgeMap = np.array(edge_map, dtype=np.uint8)
        self.update_value()
        return
        
        
    def crossover(self, mate):
        ''' random size crossover '''
#        x1, y1 = randint(0, self.N-1), randint(0, self.N-1)
#        x2, y2 = randint(0, self.N-1), randint(0, self.N-1)
#        new_edge_map1 = np.array(self.edgeMap)
#        new_edge_map1[x1:x2,y1:y2] = mate.edgeMap[x1:x2,y1:y2]
#        new_edge_map2 = np.array(mate.edgeMap)
#        new_edge_map2[x1:x2,y1:y2] = self.edgeMap[x1:x2,y1:y2]
        ''' 3*3 crossover '''
        xx, yy = randint(0, self.N-3), randint(0, self.N-3)
        new_edge_map1 = np.array(self.edgeMap)
        new_edge_map1[xx:xx+3,yy:yy+3] = mate.edgeMap[xx:xx+3,yy:yy+3]
        new_edge_map2 = np.array(mate.edgeMap)
        new_edge_map2[xx:xx+3,yy:yy+3] = self.edgeMap[xx:xx+3,yy:yy+3]

        return Individual(new_edge_map1, subregion=self.subregion, grid_size=self.N),\
               Individual(new_edge_map2, subregion=self.subregion, grid_size=self.N)
        
    def mutate(self, mutation_rate):
        global CHANGE
        ''' single-point mutation '''
        mutation_position = np.random.random([self.N, self.N])
        mutation_position[mutation_position>1-mutation_rate]=1
        mutation_position[mutation_position<=1-mutation_rate]=0
        new_edge_map = np.array(self.edgeMap)
        new_edge_map = new_edge_map + mutation_position
        new_edge_map[new_edge_map==2]=0
        if CHANGE:
            ''' multiple-point transformation mutation '''
            for k in range((self.N-2)**2):
                i=k // (self.N-2)
                j=k % (self.N-2)
                if random()>mutation_rate:
                    continue
                kernel = np.array([[0,0,1],[0,1,0],[1,0,0]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,0,1],[0,0,1],[1,1,0]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,1],[1,0,0],[1,0,0]])
                    continue
                kernel = np.array([[1,0,0],[0,1,0],[0,0,1]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[1,0,0],[1,0,0],[0,1,1]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[1,1,0],[0,0,1],[0,0,1]])
                    continue
                kernel = np.array([[0,0,1],[0,1,0],[0,1,0]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,0,0],[0,0,1],[0,1,0]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,1],[1,0,0],[0,1,0]])
                    continue
                kernel = np.array([[1,0,0],[0,1,0],[0,1,0]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,0,0],[1,0,0],[0,1,0]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[1,1,0],[0,0,1],[0,1,0]])
                    continue
                kernel = np.array([[0,1,0],[0,1,0],[0,0,1]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[0,0,1],[0,0,0]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[1,0,0],[0,1,1]])
                    continue
                kernel = np.array([[0,1,0],[0,1,0],[1,0,0]])
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[1,0,0],[0,0,0]])
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[0,0,1],[1,1,0]])
                    continue
                kernel = np.array([[0,0,1],[0,1,0],[0,1,0]]).T
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,0,0],[0,0,1],[0,1,0]]).T
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,1],[1,0,0],[0,1,0]]).T
                    continue
                kernel = np.array([[1,0,0],[0,1,0],[0,1,0]]).T
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,0,0],[1,0,0],[0,1,0]]).T
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[1,1,0],[0,0,1],[0,1,0]]).T
                    continue
                kernel = np.array([[0,1,0],[0,1,0],[0,0,1]]).T
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[0,0,1],[0,0,0]]).T
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[1,0,0],[0,1,1]]).T
                    continue
                kernel = np.array([[0,1,0],[0,1,0],[1,0,0]]).T
                if (new_edge_map[i:i+3,j:j+3]==kernel).all():
                    if randint(0,1):
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[1,0,0],[0,0,0]]).T
                    else:
                        new_edge_map[i:i+3,j:j+3] = np.array([[0,1,0],[0,0,1],[1,1,0]]).T
                    continue
            
        return Individual(new_edge_map, subregion=self.subregion, grid_size=self.N)
        
    def update_value(self):
        self.value = 0.0
        
        nb = np.ones_like(self.target)
        nb[1:self.N-1,1:self.N-1] = 0
        nb[0,0], nb[0,self.N-1], nb[self.N-1,0], nb[self.N-1,self.N-1] = 2, 2, 2, 2
        nb[0:self.N-1,0:self.N-1] = self.edgeMap[1:self.N,1:self.N] + nb[0:self.N-1,0:self.N-1]
        nb[0:self.N-1,1:self.N] = self.edgeMap[1:self.N,0:self.N-1] + nb[0:self.N-1,1:self.N]
        nb[1:self.N,0:self.N-1] = self.edgeMap[0:self.N-1,1:self.N] + nb[1:self.N,0:self.N-1]
        nb[1:self.N,1:self.N] = self.edgeMap[0:self.N-1,0:self.N-1] + nb[1:self.N,1:self.N]
        nb[0:self.N-1,:] = self.edgeMap[1:self.N,:] + nb[0:self.N-1,:]
        nb[:,0:self.N-1] = self.edgeMap[:,1:self.N] + nb[:,0:self.N-1]
        nb[1:self.N,:] = self.edgeMap[0:self.N-1,:] + nb[1:self.N,:]
        nb[:,1:self.N] = self.edgeMap[:,0:self.N-1] + nb[:,1:self.N]
        
        Ce = np.array(self.edgeMap)    # restrict number of edge pixels 
        
        Cd = np.zeros_like(self.target)    # find edge pixels
        Cd[self.edgeMap==0] = self.dissim[self.edgeMap==0]
        
        Cc = np.zeros_like(self.target)    # curvature
        Ct = np.zeros_like(self.target)    # thickness
        Cc[1:self.N-1,1:self.N-1] = 1
        Ct[1:self.N-1,1:self.N-1] = 1
        for k in range((self.N-2)**2):
            i=k // (self.N-2)
            j=k % (self.N-2)
            if self.edgeMap[i+1,j+1]==0 or nb[i+1,j+1]==0 or nb[i+1,j+1]==1:
                Cc[i+1,j+1] = 0
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[1,0,0],[0,1,0],[0,0,1]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all():
                Cc[i+1,j+1] = 0
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,1,0],[0,1,0],[0,1,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all():
                Cc[i+1,j+1] = 0
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,0,1],[0,1,0],[1,0,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all():
                Cc[i+1,j+1] = 0
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,0,0],[1,1,1],[0,0,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all():
                Cc[i+1,j+1] = 0
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[1,0,0],[0,1,0],[0,1,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all()\
            or (self.edgeMap[i:i+3,j:j+3]==kernel.T).all():
                Cc[i+1,j+1] = 0.5
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,1,0],[0,1,0],[1,0,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all()\
            or (self.edgeMap[i:i+3,j:j+3]==kernel.T).all():
                Cc[i+1,j+1] = 0.5
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,1,0],[0,1,0],[0,0,1]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all()\
            or (self.edgeMap[i:i+3,j:j+3]==kernel.T).all():
                Cc[i+1,j+1] = 0.5
                Ct[i+1,j+1] = 0
                continue
            kernel = np.array([[0,0,1],[0,1,0],[0,1,0]])
            if (self.edgeMap[i:i+3,j:j+3]==kernel).all()\
            or (self.edgeMap[i:i+3,j:j+3]==kernel.T).all():
                Cc[i+1,j+1] = 0.5
                Ct[i+1,j+1] = 0
                continue
        
        Cf = np.zeros_like(self.target)    # isolated
        Cf[(self.edgeMap-1)**2+(nb-0)**2==0] = 1
        Cf[(self.edgeMap-1)**2+(nb-1)**2==0] = 0.5
        
#        Ct = np.zeros_like(self.target)    # thickness
#        Ct[(self.edgeMap-1)**2 + ((nb-3)*(nb-4)*(nb-5)*(nb-6)*(nb-7)*(nb-8))**2==0] = 1
        
        self.value = Wd*np.sum(Cd) + Wc*np.sum(Cc) + We*np.sum(Ce)\
                     + Wf*np.sum(Cf) + Wt*np.sum(Ct)
        return
        
class Population:
    def __init__(self, size=64, crossover_rate=0.8, mutation_rate=0.01,
                       subregion=(0,0), grid_size=8):
        self.size = size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.children = []
        for i in range(size):
            self.population.append(Individual(random_init=True, 
                                              subregion=subregion,
                                              grid_size=grid_size))
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

        ''' Linear Ranking '''
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
                self.children.extend([c1, c2])
        for idx in range(len(self.children)):
            self.children.extend([self.children[idx].mutate(self.mutation_rate)])
        self.survive()
        self.population = sorted(self.population, key=lambda x: x.value)
        return

class GA_2D:
    def __init__(self, pop_size=64, crossover_rate=0.8,
                 mutation_rate=0.01, grid_size=8):
        if not N%grid_size==0:
            print('ERROR grid_size = %d'%(grid_size))
            exit(-1)
        self.grid_size = grid_size
        self.numOfGA = (N//grid_size)**2
        self.size=pop_size
        self.PP = []
        for i in range(N//grid_size):
            for j in range(N//grid_size):
                self.PP.append(Population(size=pop_size,
                                     crossover_rate=crossover_rate,
                                     mutation_rate=mutation_rate,
                                     subregion=(i,j),
                                     grid_size=grid_size))
        return
        
    def evolve(self):
        for i in range(self.numOfGA):
            self.PP[i].evolve()
        return
        
    def update_value(self):
        for i in range(self.numOfGA):
            for j in range(self.size):
                self.PP[i].population[j].update_value()
        return
                
    def show(self,generation=0):
        show_img = np.zeros_like(TARGET, dtype=np.uint8)
        for i in range(self.numOfGA):
            temp = (i//(N//self.grid_size))*self.grid_size
            sX, eX = temp, temp+self.grid_size
            temp = (i%(N//self.grid_size))*self.grid_size
            sY, eY = temp, temp+self.grid_size
            show_img[sX:eX,sY:eY] = self.PP[i].population[0].edgeMap
        show_img = show_img*255
        show_img = cv2.resize(show_img, (3*N,3*N), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('show_img',show_img)
#        cv2.imwrite('./show_img_%03d.jpg'%(generation),show_img)
        cv2.waitKey(50)
        
    def get_cost(self):
        cost = 0.0
        for i in range(self.numOfGA):
            cost = cost + self.PP[i].population[0].value
        return cost
        
    def get_numOfEdge(self):
        numOfEdge = 0.0
        for i in range(self.numOfGA):
            numOfEdge = numOfEdge + np.sum(self.PP[i].population[0].edgeMap)
        return numOfEdge
        

if __name__ == "__main__":
    maxGenerations = 100
    top_value = 1000
    Tstart = time.time()
    Cost_Set = []
    GA = GA_2D(pop_size=64, crossover_rate=0.8, mutation_rate=0.01, grid_size=8)
    for i in range(1, maxGenerations + 1):
        cost = GA.get_cost()
        print("Generation %d: %f %d"%(i, cost, GA.get_numOfEdge()))
        Cost_Set.append(cost)
        GA.evolve()
        GA.show(i)
        if i==60:
            Wf = 3
            CHANGE = True
            GA.update_value()
    Tend = time.time()
    
#    plt.plot(list(range(1,maxGenerations+1)),Cost_Set)
#    plt.ylim(0,1000)
#    plt.savefig('./image8_time_%f.png'%(Tend-Tstart))