import random
import copy
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self, id):
        self.id = id
        self.neighbors = []
        self.assignment = random.randint(0, 9)
        self.inbox = []
        self.inbox_4_mgm=[]
        self.wanted_assignment=-999
        self.wanted_savings=-999


    def calculatebestassignment(self):#checks his entire domain, domain is 0-9 . returns the best assignment 4 him
        self.inbox.sort(key=lambda x: x[0])
        #print(self.reportedvalst)
        sumcosts = self.calc_local_cost() # current
        assignment= self.assignment
        for i in range(10):
            current = 0
            for j in range(len(self.neighbors)):
                current += self.neighbors[j].costlist[i][self.inbox[j][1]]
            if (current < sumcosts):
                assignment = i
                sumcosts = current
        return assignment
    def calculatebestassignment_MGM(self):# similar but returns the savings and saves 4 himself the wanted assignment
        self.inbox.sort(key=lambda x: x[0])
        sumcosts = self.calc_local_cost() # current
        wanted_assignment= self.assignment
        savings=0
        for i in range(10):
            current = 0
            for j in range(len(self.neighbors)):
                current += self.neighbors[j].costlist[i][self.inbox[j][1]]
            if (  sumcosts-current>savings):
                wanted_assignment  = i
                savings = sumcosts-current
        self.wanted_assignment=wanted_assignment
        self.wanted_savings=savings


    def take_your_mail(self,mail_box):
        self.inbox=copy.deepcopy(mail_box[self.id])
    def take_your_mail_MGM(self,mail_box):
        self.inbox_4_mgm=copy.deepcopy(mail_box[self.id])

class Neighbor:# index and cost list, no pointers as required
    def __init__(self, index, costlist):
        self.index = index
        self.costlist = costlist

def create_matrix(k):#neighbors matrix
    matrix = [[0] * 30 for _ in range(30)]

    for i in range(30):
        for j in range(i + 1, 30):
            if (random.random() <= k and j != i):
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


def generate_symmetric_list():#cost matrix, transformable (linear algebra)
    symmetric_list = [[0] * 10 for _ in range(10)]
    for i in range(10):
        for j in range(i, 10):
            value = random.randint(0, 100)
            symmetric_list[i][j] = value
            symmetric_list[j][i] = value
    return symmetric_list

def fireup(k):
    #create everything to start, fitting both mgm and dsa_c
    numagents=30
    matrix = create_matrix(k)
    matrix_30_30 = [[copy.deepcopy(generate_symmetric_list()) for _ in range(30)] for _ in range(30)]
    mail_box= [[] for _ in range(30)]
    agents = []
    for i in range(numagents):
        agents.append(Agent(i))
    for i in range(numagents):
        for j in range(numagents):
            if (matrix[i][j] == 1):
                agents[i].neighbors.append(Neighbor(j, copy.deepcopy(matrix_30_30[min(i,j)][max(i,j)])))
    for i in range(numagents):
        agents[i].neighbors.sort(key=lambda x: x.index)
    for i in range(numagents):
        for j in range(len(agents[i].neighbors)):
            mail_box[agents[i].neighbors[j].index].append([agents[i].id,agents[i].assignment])
    cost=0
    for a in  range(len(agents)):
       agents[a].neighbors.sort(key=lambda x: x.index)
       mail_box[a].sort(key=lambda x: x[0])
       for j in range(len(agents[a].neighbors)):
           cost += agents[a].neighbors[j].costlist[agents[a].assignment][mail_box[a][j][1]]
    return [agents,cost,mail_box]

def dsa_c(p,k,what,num_iterations):
    random.seed(246)
    num_trials = 30
    cost_lists = []
    for _ in range(num_trials):
        print(_)
        starter=fireup(k)
        agents=copy.deepcopy(starter[0])
        cost_list = []
        cost_list.append(starter[1])
        mail_box=copy.deepcopy(starter[2])
        for r in range(num_iterations):
            for a in agents:
                a.take_your_mail(mail_box)
            mail_box = [[] for _ in range(30)]
            cost = 0
            for a in agents:#bird's eye view of the costs
                for j in range(len(a.neighbors)):
                    cost += a.neighbors[j].costlist[a.assignment][agents[a.neighbors[j].index].assignment]
            cost_list.append(cost)
            for a in agents:
                i = a.calculatebestassignment()
                if (random.uniform(0, 1) < p):
                    a.assignment = i
                for j in range(len(a.neighbors)):#always, even if didnt change assignment
                    mail_box[a.neighbors[j].index].append([a.id, a.assignment])
        cost_lists.append(cost_list)


    average_costs = [sum(x) / num_trials for x in zip(*cost_lists)]
    if(what==1):#modularity of the code
        return range(num_iterations+1), average_costs
    elif(what==2):
        return average_costs[len(average_costs)-1]

def mgm(k,num_iterations):
    random.seed(246)
    num_trials = 30
    cost_lists = []
    for _ in range(num_trials):
        print(_)
        starter=fireup(k)
        agents=copy.deepcopy(starter[0])
        cost_list = []
        cost_list.append(starter[1])
        mail_box=copy.deepcopy(starter[2])
        for r in range(num_iterations):
            cost = 0
            for a in agents:
                for j in range(len(a.neighbors)):
                    cost += a.neighbors[j].costlist[a.assignment][agents[a.neighbors[j].index].assignment]
            cost_list.append(cost)
            for a in agents:
                a.take_your_mail(mail_box)

            mail_box = [[] for _ in range(30)]
            for a in agents:
                a.calculatebestassignment_MGM()
                for j in range(len(a.neighbors)):#now they update how much they would save
                    mail_box[a.neighbors[j].index].append([a.id,a.wanted_savings])
            for a in agents:
                a.take_your_mail_MGM(mail_box)
            mail_box = [[] for _ in range(30)]
            for a in agents:
                should_i_switch=True
                for i in range(len(a.inbox_4_mgm)):
                    if(a.inbox_4_mgm[i][1]>a.wanted_savings or (a.inbox_4_mgm[i][1]==a.wanted_savings and a.inbox_4_mgm[i][0]<a.id) ):#arbitrary, to prevent both agents from switching
                        should_i_switch=False
                if(should_i_switch):
                    a.assignment=a.wanted_assignment
                for j in range(len(a.neighbors)):
                    mail_box[a.neighbors[j].index].append([a.id, a.assignment])

        cost_lists.append(cost_list)
    average_costs = [sum(x) / num_trials for x in zip(*cost_lists)]
    return range(num_iterations+1), average_costs


if __name__ == '__main__':

    if __name__ == '__main__':
        #for graph 1
        p_values = np.arange(0, 1.01, 0.05)  # generate p values from 0 to 1 with step 0.25
        k = 0.2  # fixed k value
        what = 2  # fixed what value
        num_iterations=1000

        results = []
        for p in p_values:
            res = dsa_c(p, k, what,num_iterations)
            results.append(res)

        plt.figure()
        plt.plot(p_values, results, 'o-', label='dsa_c results')
        plt.xlabel('p value')
        plt.ylabel('dsa_c result')
        plt.title('dsa_c result as a function of p value, K=0.2')
        plt.legend(loc='upper right')
        plt.show()
        # for graph 2
        iterations, avg_costs_02 = dsa_c(0.2, 0.2, 1,300)
        iterations, avg_costs_07 = dsa_c(0.7, 0.2, 1,300)
        iterations, avg_costs_MGM = mgm(0.2,300)
        plt.figure()
        plt.plot(iterations, avg_costs_02, label='p=0.2')
        plt.plot(iterations, avg_costs_07, label='p=0.7')
        plt.plot(iterations, avg_costs_MGM, label='mgm')
        plt.xlabel('Iteration')
        plt.ylabel('Average Cost')
        plt.title('Average Cost as a function of iteration, K=0.2, Graph No 2')
        plt.legend(loc='upper right')
        plt.show()
        # for graph 3
        iterations, avg_costs_02 = dsa_c(0.2, 0.7, 1,300)
        iterations, avg_costs_07 = dsa_c(0.7, 0.7, 1,300)
        iterations, avg_costs_MGM = mgm(0.7,300)
        plt.figure()
        plt.plot(iterations, avg_costs_02, label='p=0.2')
        plt.plot(iterations, avg_costs_07, label='p=0.7')
        plt.plot(iterations, avg_costs_MGM, label='mgm')
        plt.xlabel('Iteration')
        plt.ylabel('Average Cost')
        plt.title('Average Cost as a function of iteration, K=0.7, Graph No 3')
        plt.legend(loc='upper right')





