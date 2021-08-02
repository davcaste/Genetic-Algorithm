import Reporter
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
# Modify the class name to match your student number.
class r0834601:

    def __init__(self):

        self.population_size = 100
        self.new_population = 100
        self.city_score_dict = {}
        self.ten_scores = np.ones(100) * 10000000000000000000000000000000000
        self.ten_scores[0] = 10000000000000000000000000000000001
        self.min_ever = 1000000000000000000000001
        pass

    # Critical distance that is used for a type of crossover (new_crossover)
    def calculate_crit_dist(self, B = 2):
        N = self.num_of_cities
        factor = 1 / (B * (N - 1))
        crit_dist = []
        for i in range(self.num_of_cities):
            sum_dist = np.sum(self.distanceMatrix[i, :])
            crit_dist.append(factor * sum_dist)
        return crit_dist

    # Simple swap mutation
    def mutate(self, scores):
        new_l = copy.deepcopy(self.list_of_perm)

        for k, perm in enumerate(new_l):
            if np.random.random() < 0.2:  # Trigger a mutation with a 10% chance
                i = np.random.randint(0, len(perm))
                j = np.random.randint(0, len(perm))
                perm[[i, j]] = perm[[j, i]]
                self.list_of_perm = np.append(self.list_of_perm, [perm], axis=0)
                scores.append(self.calculate_score(perm))
        return scores

    # Another mutation that swap two sequences instead of two city
    def mutation(self, scores):
        new_l = copy.deepcopy(self.list_of_perm)
        for i, perm in enumerate(new_l):
            supp = np.zeros(len(perm), dtype=int)
            if np.random.random() < 0.1:  # Trigger a mutation with a 10% chance
                i = np.random.randint(0, len(perm) - 2)
                j = np.random.randint(i, len(perm) - 1)
                l = np.random.randint(j, len(perm) - 1)
                k = np.random.randint(l, len(perm))

                supp[0:i] = perm[0:i]
                supp[i:(i + k - l)] = perm[l:k]
                supp[(i + k - l):(i + k - j)] = perm[j:l]

                supp[(i + k - j):(k)] = perm[i:j]

                supp[k:-1] = perm[k:-1]
                supp[-1] = perm[-1]
                perm = np.array(supp)
            self.list_of_perm = np.append(self.list_of_perm, [perm], axis=0)
            scores.append(self.calculate_score(perm))
        return scores

    # Function that calculate the score in an optimized way
    def calculate_score(self, perm):
        support = [self.distanceMatrix[perm[z], perm[z + 1]] for z in range(len(perm) - 1)]
        support.append(self.distanceMatrix[perm[-1], perm[0]])
        accumulator = np.sum(support)

        return accumulator

    # Edge recombination crossover (not used due to inefficiency and bad results)
    def edge_recombination_crossover(self, parent1, parent2):
        adj_mat = self.compute_adjancy(parent1, parent2)

        starting = np.random.choice(parent1)
        child1 = [starting]

        flag = False
        for i in range(len(parent1) - 1):
            min_len = 29
            poss_next = []
            if flag == False:
                for values in adj_mat[starting]:

                    # Aggiungere random se hanno tutti la lunghezza uguale

                    adj_mat[values].remove(starting)
                    supp = len(adj_mat[values])
                    if supp == min_len:
                        poss_next.append(values)
                    if supp < min_len:
                        poss_next = [values]
                        min_len = supp

                key_min_len = np.random.choice(poss_next)

            else:
                adj_mat.pop(starting)
                key_min_len = np.random.choice(list(adj_mat.keys()))
                adj_mat[starting] = {1}

                min_len = len(adj_mat[key_min_len])
                flag = False

            if min_len == 0:
                flag = True

            adj_mat.pop(starting)
            starting = key_min_len
            child1.append(key_min_len)

        return child1

    # Computation of adjancy matrix for edge recombination
    def compute_adjancy(self, parent1, parent2):
        adj_mat = {}
        for i, j in enumerate(parent1):
            ind = list(parent2).index(j)
            if (i == 0) and (ind == 0):
                adj_mat[j] = {parent1[-1], parent1[1], parent2[-1], parent2[1]}
                continue
            elif (i == 0) and (ind == len(parent1) - 1):
                adj_mat[j] = {parent1[-1], parent1[1], parent2[-2], parent2[0]}
            ind = list(parent2).index(j)
            if i == len(parent1) - 1:
                break
            if ind == len(parent1) - 1:
                adj_mat[j] = {parent1[i - 1], parent1[i + 1], parent2[ind - 1], parent2[0]}
                continue
            adj_mat[j] = {parent1[i - 1], parent1[i + 1], parent2[ind - 1], parent2[ind + 1]}
        ind = list(parent2).index(j)
        if ((ind == len(parent1) - 1) and (i != len(parent1) - 1)):
            adj_mat[j] = {parent1[i - 1], parent1[i + 1], parent2[ind - 1], parent2[0]}
        elif ((ind == len(parent1) - 1) and (i == len(parent1) - 1)):
            adj_mat[j] = {parent1[i - 1], parent1[0], parent2[ind - 1], parent2[0]}

        else:
            adj_mat[j] = {parent1[-2], parent1[0], parent2[ind - 1], parent2[ind + 1]}
        return adj_mat

    # Order crossover (very efficient but not used due to bad results)
    def order_crossover(self, parent1, parent2):
        child = []
        childP1 = []

        geneA = np.random.randint(1, len(parent1))
        geneB = np.random.randint(1, len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])

        childP2 = [item for item in parent2 if item not in childP1]
        child1 = []
        childP11 = []

        for i in range(startGene):
            child1.append(childP2[i])
        for i in childP1:
            child1.append(i)
        for i in range(startGene, len(childP2)):
            child1.append(childP2[i])

        for i in range(startGene, endGene):
            childP11.append(parent2[i])

        childP2 = [item for item in parent1 if item not in childP11]

        for i in range(startGene):
            child.append(childP2[i])
        for i in childP11:
            child.append(i)
        for i in range(startGene, len(childP2)):
            child.append(childP2[i])

        return child, child1

    # Sequential contructive crossover (good result in an acceptable amount of time
    def scx(self, parent1, parent2):
        starting = random.choice(parent1)
        dm = copy.deepcopy(self.distanceMatrix)
        path = [starting]
        for i in range(self.num_of_cities - 1):
            ind1 = parent1.tolist().index(starting)
            ind2 = parent2.tolist().index(starting)
            # if (0 < ind1 < (self.num_of_cities - 1)) and (0 < ind2 < (self.num_of_cities - 1)):
            # check11, check12 = ind1 - 1, ind1 + 1
            # check21, check22 = ind2 - 1, ind2 + 1
            # else:
            check11, check12 = ind1 - 1, ind1 + 1
            check21, check22 = ind2 - 1, ind2 + 1
            if ind1 == 0:
                check11 = self.num_of_cities - 1
            elif ind1 == (self.num_of_cities - 1):
                check12 = 0
            if ind2 == 0:
                check21 = self.num_of_cities - 1
            elif ind2 == (self.num_of_cities - 1):
                check22 = 0
            supp = [check11, check12, check21, check22]
            a = [dm[starting, parent1[check11]], dm[starting, parent1[check12]], dm[starting, parent2[check21]],
                 dm[starting, parent2[check22]]]
            if a == [self.maxim * 3, self.maxim*3, self.maxim*3, self.maxim*3]:
                old = starting
                dm[starting, starting] = self.maxim*3
                starting = dm[starting, :].tolist().index(min(dm[starting, :]))
                path.append(starting)
                dm[:, old] = self.maxim*3
                dm[old, :] = self.maxim*3
                continue

            minimum = a.index(min(a))
            dm[starting, :] = self.maxim*3
            dm[:, starting] = self.maxim*3

            next_one = supp[minimum]
            if (minimum == 0) or (minimum == 1):
                starting = parent1[next_one]
            else:
                starting = parent2[next_one]
            path.append(starting)
        d = {}
        return path

    # Adaptation of scx (perform slightly better but a bit slower)
    def new_crossover(self, parent1, parent2):

        dm = copy.deepcopy(self.distanceMatrix)

        # New concept: calculation of the critical distance (offline) and then using this parameter to decide if to take
        # one of the neighbour in the parent array or another city with minimum distance

        crit_dist = self.crit_dist

        starting = np.random.choice(parent1)
        path = [starting]

        for i in range(self.num_of_cities - 1):
            dm[:, starting] = self.maxim*3
            ind1 = parent1.tolist().index(starting)
            ind2 = parent2.tolist().index(starting)
            j11 = ind1+1
            j12 = ind1-1
            j21 = ind2+1
            j22 = ind2-1
            if ind1 == (self.num_of_cities-1):
                j11 = 0
            elif ind1 == 0:
                j12 = self.num_of_cities-1
            if ind2 == (self.num_of_cities-1):
                j21 = 0
            elif ind2 == 0:
                j22 = self.num_of_cities-1
            a = [dm[starting,parent1[j11]], dm[starting,parent1[j12]], dm[starting,parent2[j21]], dm[starting,parent2[j22]]]
            par_ind = [j11,j12,j21,j22]
            d_min = min(a)
            ind_min = a.index(d_min)
            if d_min < crit_dist[starting]:
                if ind_min == 0 or ind_min == 1:
                    starting = parent1[par_ind[ind_min]]
                else:
                    starting = parent2[par_ind[ind_min]]
            else:
                starting = dm[starting, :].tolist().index(min(dm[starting, :]))
            path.append(starting)
        for i in range(self.num_of_cities):
            if i not in path:
                return 0
        return path

    # K-tournament selection work well and fast
    def k_tournament_selection(self, parent_array, scores, candidates=5):

        selected = random.sample(range(self.population_size), candidates)

        best1 = 100000000000000000000000
        best2 = 1000000000000000000000000
        index1, index2 = 0, 1
        for i in range(len(selected)):
            if scores[selected[i]] < best1:
                best1 = scores[selected[i]]
                index1 = selected[i]
            else:
                if scores[selected[i]] < best2:
                    best2 = scores[selected[i]]
                index2 = selected[i]

        return parent_array[index1], parent_array[index2]

    def roulette_wheel(self, population, scores):
        maximum = sum(scores)
        selection_probs = [scores[c] / maximum for c in range(len(population))]

        return population[np.random.choice(len(population), p=selection_probs)]


    # (λ+μ) elimination
    def elimination(self, scores):
        dictionary = {}
        for i, score in enumerate(scores):
            dictionary[score] = self.list_of_perm[i]
        initial_dict = copy.deepcopy(dictionary)
        a = scores

        N = self.population_size

        b = np.array(a[:])
        lop = []
        los = []
        for i in range(N):
            if len(dictionary) == 0:
                dictionary = copy.deepcopy(initial_dict)
            min_k = min(dictionary.keys())
            los.append(min_k)
            lop.append(dictionary[min_k])
            dictionary.pop(min_k)

        self.list_of_perm = np.array(lop)
        scores = los

        return scores

    # Stopping criterion: obtain 100 times the same scores
    def stopping_criterion(self, scores):
        best_score = min(scores)
        self.ten_scores[-1] = best_score
        self.ten_scores = np.roll(self.ten_scores, 1)

        new_min = min(self.ten_scores)
        if new_min < self.min_ever:
            self.min_ever = new_min
        if list(self.ten_scores).count(self.ten_scores[0]) == len(self.ten_scores):  # and new_min == self.min_ever:
            return False
        else:
            return True

    # def heuristic_initialization(self):
    #     initial_path = [0]
    #     dm = copy.deepcopy(self.distanceMatrix)
    #     dm[0, 0] = 1000000
    #     prox = np.argmin(dm[0])
    #     initial_path.append(prox)
    #     dm[0, prox] = 1000000
    #     dm[prox, 0] = 1000000
    #     for i in range(1, self.num_of_cities - 1):
    #         prev = prox
    #         dm[prev, prev] = 1000000
    #         prox = np.argmin(dm[prox])
    #         dm[prox, (initial_path)] = 10000000
    #         initial_path.append(prox)
    #         dm[prev, prox] = 1000000
    #         dm[prox, prev] = 1000000
    #     return initial_path

    # Heuristic initialization that select a random starting city and pick as next one the closer one (diversity
    # guaranteed by the different starting city that result in different paths)
    def heuristic_initialization_1(self):
        starting = random.choice(self.list_cities)
        initial_path = [starting]
        dm = copy.deepcopy(self.distanceMatrix)
        dm[starting, starting] = self.maxim*3
        prox = np.argmin(dm[starting])
        initial_path.append(prox)
        dm[starting, prox] = self.maxim*3
        dm[prox, starting] = self.maxim*3
        for i in range(1, self.num_of_cities - 1):
            prev = prox
            dm[prev, prev] = self.maxim*3
            prox = np.argmin(dm[prox])
            dm[prox, initial_path] = self.maxim * 3
            initial_path.append(prox)
            dm[prev, prox] = self.maxim*3
            dm[prox, prev] = self.maxim*3
        return initial_path

    # Probabilistic initialization (not used because too slow)
    def initialization(self):
        initial_path = [0]
        all_cities = list(range(len(self.distanceMatrix)))
        dm = copy.deepcopy(self.distanceMatrix)
        dm[0, 0] = self.maxim * 3
        s = 0
        prob = np.zeros(self.num_of_cities)
        for i in range(1, self.num_of_cities):
            sum_dist = sum(1 / dm[s])
            subtract = sum(1 / dm[s, initial_path])
            sum_dist -= subtract
            for j in range(len(prob)):
                if j in initial_path:
                    prob[j] = 0
                    continue
                prob[j] = (1 / dm[s, j]) / sum_dist
            selected = all_cities[np.random.choice(len(all_cities), p=prob)]
            initial_path.append(selected)
            s = selected
            dm[s, s] = self.maxim*3
            dm[s, initial_path] = self.maxim*3
        return initial_path

    # Revisited (approximated) probabilitic initialization in which the rows of the distance matrix are divided in
    # percentiles and then the probability of selecting a city decrease with the increasing of the percentile in which
    # they are
    def other_initialization(self):

        initial_path = [0]
        dm = copy.deepcopy(self.distanceMatrix)
        s = 0
        dm[0, 0] = self.maxim*3
        for i in range(1, self.num_of_cities):
            percentage = np.percentile(self.distanceMatrix[s, :], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            poss = []
            for j in percentage:
                l = self.distanceMatrix[s, dm[s, :] <= j]
                poss.append(l)
            flag = False
            chosen = random.randint(0, len(percentage) - 1)
            while not flag:
                if len(poss[chosen]) == 0:
                    chosen += 1
                else:
                    flag = True
            selected = random.choice(poss[chosen])
            ind = list(dm[s, :]).index(selected)
            selected = ind
            initial_path.append(selected)
            s = selected
            dm[s, s] = self.maxim*3
            dm[s, initial_path] = self.maxim*3
        return initial_path


    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        self.reporter = Reporter.Reporter(self.__class__.__name__)        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        dm = np.isinf(self.distanceMatrix)
        self.distanceMatrix[dm] = 1
        self.maxim = np.max(self.distanceMatrix)
        self.distanceMatrix[dm] = self.maxim * 2

        # Convert the matrix in int --> faster and small approximation

        # dist_int = self.distanceMatrix.round()

        # self.distanceMatrix = dist_int[:].astype('int')
        # self.distanceMatrix = self.distanceMatrix-2
        # print('modified')

        # Comment till here if want to use float
        file.close()
        self.num_of_cities = len(self.distanceMatrix)

        self.crit_dist = self.calculate_crit_dist()
        li = []
        self.list_cities = list(range(self.num_of_cities))
        timeLeft = 300

        # Initializartion
        for i in range(self.population_size):
            # if i%3 == 0:
            # Random initialization: work very good, better result, but for very big dataset doesn't compute enough cycles
            # perm = [x for x in range(0, self.num_of_cities)]
            # np.random.shuffle(perm)
            # perm.insert(0, 0)
            # elif i%3 ==1:
            # if len(possible_starting) == 0:
            #     possible_starting = list(range(self.num_of_cities))

            # Heuristic: more or less same results as random initialization but starts already in a good position
            perm = self.heuristic_initialization_1()
            # possible_starting.remove(initial)
            # elif i%3 == 2:

            # Probabilistic initialization: works worse than the others and it's slow
            # perm = self.other_initialization()
            # perm = self.initialization()

            # print(i, self.calculate_score(perm), perm)
            li.append(perm)

        self.list_of_perm = np.array(li)
        myl = list(self.list_of_perm)
        scores = [self.calculate_score(i) for i in myl]

        # Your code here.

        flag = True
        iterations = 0
        TimeLeft = []
        BestObjective = []
        MeanObjective = []

        # Optimization:
        while flag:

            iterations += 1
            parent_population = copy.deepcopy(self.list_of_perm)
            scores_copy = copy.deepcopy(scores)


            for i in range(self.new_population):

                parent1, parent2 = self.k_tournament_selection(parent_population, scores_copy)
                # parent1 = self.roulette_wheel(parent_population, scores_copy)
                # parent2 = self.roulette_wheel(parent_population, scores_copy)
                # child1 = self.new_crossover(parent1,parent2)
                child1 = self.scx(parent1, parent2)
                # child2 = self.scx(parent1, parent2)
                # child1, child2 = self.order_crossover(parent1, parent2)
                # child1 = self.edge_recombination_crossover(parent1, parent2)

                self.list_of_perm = np.append(self.list_of_perm, [child1], axis=0)
                scores.append(self.calculate_score(child1))

                # self.list_of_perm = np.append(self.list_of_perm, [child2], axis=0)
                # scores.append(self.calculate_score(child2))

            # Error here

            scores = self.mutate(scores)

            scores = self.mutation(scores)

            scores = self.elimination(scores)

            # print(scores)

            flag = self.stopping_criterion(scores)

            # Calculate the new objective function values for the population after elimination
            meanObjective = np.mean(scores)
            bestObjective = min(scores)
            # print(iterations, bestObjective)
            location = scores.index(bestObjective)
            bestSolution = self.list_of_perm[location]

            TimeLeft.append(300 - timeLeft)
            BestObjective.append(bestObjective)
            MeanObjective.append(meanObjective)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            if timeLeft < 0:
                break
        myl = list(self.list_of_perm)
        scores = [self.calculate_score(i) for i in myl]
        min_value = min(scores)
        # horiz_line_data = np.array([27154 for i in range(len(TimeLeft))])
        # print(horiz_line_data)
        # print(vec1, vec)
        # plt.figure()
        # plt.plot(TimeLeft, horiz_line_data,'g--', label='Global Optima')
        # plt.show()
        plt.figure()
        fig, ax = plt.subplots()
        plt.xlabel('Time')
        plt.ylabel('Path length')
        ax.plot(TimeLeft,BestObjective, label = 'Best Objective')
        ax.plot(TimeLeft,MeanObjective, label = 'Mean Objective')
        # plt.plot(TimeLeft, [27154] * len(TimeLeft),'g--', label='Global Optima')
        # plt.plot(TimeLeft, [30350] * len(TimeLeft), 'r--', label='Greedy heuristic')
        ax.hlines(113683.58,xmin=0,xmax=300-timeLeft,colors='r', linestyles='dashed',label='Greedy heuristic')
        ax.hlines(95300, xmin=0, xmax=300 - timeLeft, colors= 'g', linestyles='dashed',label='Global optima')
        ax.legend(loc="upper right")
        plt.show()

        # data1 = BestObjective
        # data2 = MeanObjective
        #
        # fig, ax1 = plt.subplots()
        #
        # color = 'tab:red'
        # ax1.set_xlabel('TimeSpent')
        # ax1.set_ylabel('Best objective', color=color)
        # ax1.plot(TimeLeft, data1, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # color = 'tab:blue'
        # ax2.set_ylabel('MeanObjective', color=color)  # we already handled the x-label with ax1
        # ax2.plot(TimeLeft, data2, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        #
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

        min_index = scores.index(min_value)
        # print('min score: ', min_value)
        # print('best path: ', np.array(self.list_of_perm[min_index]))
        # print('score of the path: ', self.calculate_score(self.list_of_perm[min_index]))
        # print(bestSolution)
        # print(bestObjective)
        # print(self.calculate_score(bestSolution))
        return timeLeft, bestObjective, bestSolution, meanObjective


x = r0834601()
TL = []
BO = []
MO = []
for i in range(10):
    tl, bo, bs, mo = x.optimize(os.path.join("Data","tour29.csv"))
    TL.append(300-tl)
    BO.append(bo)
    MO.append(mo)
    print(i)
    print(300 - tl, bo,bs,mo)
mu = np.mean(BO)
sigma = np.std(BO)
minimum = np.min(BO)
maximum = np.max(BO)
mut = np.mean(TL)
sigmat = np.std(TL)
minimumt = np.min(TL)
maximumt =np.max(TL)
mumo = np.mean(MO)
sigmamo = np.std(MO)
minimummo = np.min(MO)
maximummo =np.max(MO)
print(mu, sigma, minimum, maximum, mut, sigmat, minimumt, maximumt,mumo, sigmamo, minimummo, maximummo)
plt.figure()
plt.hist(BO,align = 'left',rwidth=1)
plt.title('Best objective')
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
plt.figure()
plt.hist(MO,align = 'left',rwidth=1)
plt.title('Mean objective')
plt.show()

plt.figure()
plt.hist(TL, align = 'left',rwidth=1)
plt.title('Time spent')
plt.show()
plt.figure()
plt.plot(range(len(BO)), BO)
plt.show()
plt.figure()
plt.plot(range(len(TL)), TL)
plt.show()
plt.figure()
plt.plot(range(len(MO)), MO)
plt.show()
plt.figure()
plt.plot(TL,BO)
plt.show()


# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = BO
data2 = TL

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('runs')
ax1.set_ylabel('Best objective', color=color)
ax1.plot(range(len(TL)), data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Time', color=color)  # we already handled the x-label with ax1
ax2.plot(range(len(TL)), data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#
# fig = plt.figure()
# host = fig.add_subplot(111)
#
# par1 = host.twinx()
# par2 = host.twinx()
#
# host.set_xlim(0, 2)
# host.set_ylim(0, 2)
# par1.set_ylim(0, 4)
# par2.set_ylim(1, 65)
#
# host.set_xlabel("Distance")
# host.set_ylabel("Density")
# par1.set_ylabel("Temperature")
# par2.set_ylabel("Velocity")
#
# color1 = plt.cm.viridis(0)
# color2 = plt.cm.viridis(0.5)
# color3 = plt.cm.viridis(.9)
#
# p1, = host.plot([0, 1, 2], [0, 1, 2], color=color1,label="Best objective")
# p2, = par1.plot([0, 1, 2], [0, 3, 2], color=color2, label="Temperature")
# p3, = par2.plot([0, 1, 2], [50, 30, 15], color=color3, label="Velocity")
#
# lns = [p1, p2, p3]
# host.legend(handles=lns, loc='best')
#
# # right, left, top, bottom
# par2.spines['right'].set_position(('outward', 60))
# # no x-ticks
# par2.xaxis.set_ticks([])
# # Sometimes handy, same for xaxis
# #par2.yaxis.set_ticks_position('right')
#
# host.yaxis.label.set_color(p1.get_color())
# par1.yaxis.label.set_color(p2.get_color())
# par2.yaxis.label.set_color(p3.get_color())
#
# plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')
#
# # print(300 - tl, bo,bs,mo)
#
# # x_prime = [ 4625.99359593,  4516.48204136,  3438.03917589,  3367.81660505,
# #   3478.19708327,  3790.78105962,  2577.64492884,  2405.34936733,
# #   1955.57024317,  1661.74683827,  1585.89540765,  1178.64167602,
# #    643.91718955,  -490.74929332, -1023.1887561,  -1809.87646422,
# #  -1534.83539049, -1524.50709244, -1562.24620604, -2247.57011217,
# #  -2244.03089109, -2141.20748506, -2244.92311406, -2850.91994678,
# #  -2909.66883881, -2627.7588143,  -2915.88966977, -2732.26547694,
# #  -2710.43766039]
# # y_prime = [ 2995.75179687,  2901.47703114,  -963.51569948,   176.3523483,
# #    916.44415151,  2458.58313939,  -717.07803552,   470.46095103,
# #  -1009.46401554,  2174.54347127,  2312.46095467,   939.82667525,
# #   -230.57156864,  -937.33908642,  1331.2591683,  -2470.96665053,
# #   -306.39995829,   368.38720577,   789.52164788, -1321.7346013,
# #    103.27406953,   521.96590406,   278.16253844, -3117.57689445,
# #  -2494.46729764, -1092.06586015, -2454.90590279,  -470.89943554,
# #    120.51395288]
#
# # plt.scatter(x_prime,y_prime)
# # plt.legend(('Initial positions','Final positions'))
# # plt.title('AGV positions')
# # plt.show()
# # plt.figure()
#
