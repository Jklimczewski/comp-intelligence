import pygad
import numpy
import random
import time
# f = open("projekt/alg2_srednie.txt", "a")

inputs = [
    [1, 2, 3, 3, 5, 6],
    [2, 3, 6, 7, 1, 9],
    [11, 2, 13, 4, 11, 7],
    [3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [2, 67, 34, 10, 86, 6, 42, 22, 55, 19, 71, 30],
    [18, 25, 70, 41, 9, 53, 89, 9, 36, 74, 17, 87],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [98, 97, 92, 90, 87, 83, 79, 73, 67, 61, 52, 47, 39, 33, 26, 21, 15, 12, 9, 5, 3],
    [23, 64, 6, 18, 83, 52, 5, 30, 89, 14, 50, 1, 27, 72, 11, 95, 2, 17, 33, 77, 7]
]
my_input = inputs[random.randint(0, 2)]

gene_space = list(range(int(len(my_input)/3)))

def fitness_func(solution, solution_idx):
    def check_matrix(matrix):
        lengths = set(len(row) for row in matrix)
        return len(lengths) == 1
    
    solution_spread = [[] for i in range(len(gene_space))]
    for i in range(len(solution)):
        solution_spread[int(solution[i])].append(my_input[i])

    if check_matrix(solution_spread):
        results_sums = []
        for tab in solution_spread:
            results_sums.append(numpy.sum(tab))
        
        max_diff = max(results_sums) - min(results_sums)
        return -max_diff
    else:
        return -1000

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 500
num_genes = len(my_input)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 250
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = int(100/len(my_input)) + 2

    
start=time.time()

ga_instance = pygad.GA(gene_space=gene_space,
                    num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_function,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_percent_genes=mutation_percent_genes,
                    stop_criteria=["reach_0"],
                    )

ga_instance.run()

end=time.time()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
# f.write(str(solution_fitness) + " " + str(end-start)+ "\n")
# f.close()

solution_spread = [[] for i in range(len(gene_space))]
for i in range(len(solution)):
    solution_spread[int(solution[i])].append(my_input[i])
            
print("Parameters of the best solution : {solution}".format(solution=solution))
print("The best solution : {solution_spread}".format(solution_spread=solution_spread))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()