import sys
import random
import matplotlib.pyplot as plt
from tsp_util.TSPData import TSPData 

class GeneticAlgorithm:

    """
    Constructs a new 'genetic algorithm' object.
    @param generations: the amount of generations.
    @param pop_size: the population size.
    """
    def __init__(self, generations, pop_size, seed=None):
        self.generations = generations
        self.pop_size = pop_size
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.elitism_rate = 0.1
        self.population = []
        self.best = [None, sys.maxsize]
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
    def initialize_population(self, distances, start_distances, end_distances):
        number_of_products = len(distances)
        for _ in range(self.pop_size):
            chromosome = random.sample(range(number_of_products), number_of_products)
            fitness = self.calculate_fitness(chromosome, distances, start_distances, end_distances)
            self.population.append((chromosome, fitness))
        
    def select_parents(self):
        parents = random.sample(self.population, k=5)
        parents.sort(key=lambda x: x[1], reverse=True)
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        offspring = [-1] * size
        start, end = sorted(random.sample(range(size), 2))

        # Copy a slice from the first parent:
        offspring[start:end+1] = parent1[start:end+1]

        # Fill the rest with the genes from the second parent in the order they appear,
        # skipping duplicates
        current_pos = (end + 1) % size
        for gene in parent2:
            if gene not in offspring:
                offspring[current_pos] = gene
                current_pos = (current_pos + 1) % size
        return offspring

    def mutate(self, individual):
        p1, p2 = random.sample(range(len(individual)), 2)
        temp = individual[p1]
        individual[p1] = individual[p2]
        individual[p2] = temp
        return individual

    def calculate_fitness(self, individual, distances, start_distances, end_distances):
        fitness = start_distances[individual[0]]
        for i in range(len(individual) - 1):
            product_idx = individual[i]
            next_product_idx = individual[i+1]
            fitness += distances[individual[i]][individual[i+1]]
        fitness += end_distances[individual[len(individual) - 1]]
        fitness += len(individual)
        return fitness
    
    def get_best_solution(self):
        # self.population.sort(key=lambda x: x[1], reverse=True)
        # return self.population[0]
        return self.best
    
    def set_best_solution(self, best_solution):
        self.best = best_solution
    
    """
    This method should solve the TSP.
    @param tsp_data: the data describing the problem.
    @param plot_avg: Boolean indicating wether to plot avg fitness over generations
    @param plot_best: Boolean indicating wether to plot best fitness over generations
    @return the optimized product sequence.
    """
    def solve_tsp(self, tsp_data: TSPData, plot_avg=False, plot_best=False):
        distances = tsp_data.get_distances()
        start_distances = tsp_data.get_start_distances()
        end_distances = tsp_data.get_end_distances()
        
        self.initialize_population(distances, start_distances, end_distances)  
        
        elitism_count = int(self.pop_size * self.elitism_rate)
        
        average_fitness_over_generations = []
        best_fitness_over_generations = []
        
        for gen in range(self.generations):
            # Elitism
            sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)
            elites = sorted_population[:elitism_count]
            
            new_population = elites[:]
            
            # Selection
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parent1[0], parent2[0])
                else:
                    offspring = parent1[0]  # or parent2
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self.mutate(offspring)
                
                # Fitness Calculation
                offspring_fitness = self.calculate_fitness(offspring, distances, start_distances, end_distances)
                
                # Add offspring to the new population
                new_population.append((offspring, offspring_fitness))
            
            # Your GA may also include a condition to break early if you meet a certain criterion

            self.population = new_population
            
            current_average_fitness = sum(individual[1] for individual in self.population) / self.pop_size
            current_best_fitness = min(self.population, key=lambda x: x[1])

            average_fitness_over_generations.append(current_average_fitness)
            best_fitness_over_generations.append(current_best_fitness[1])
            
            if (current_best_fitness[1] < self.get_best_solution()[1]):
                self.set_best_solution(current_best_fitness)
                
        if plot_avg or plot_best:

            plt.figure(figsize=(12, 6))

            if plot_avg:
                plt.plot(average_fitness_over_generations, label='Average Fitness', color='blue')

            if plot_best:
                plt.plot(best_fitness_over_generations, label='Best Fitness', color='red')

            # Adding titles and labels
            plt.title(f'Fitness over Generations - generations : {self.generations} - population size {self.pop_size}')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()

            # Show plot
            plt.show()
        
        return self.get_best_solution()