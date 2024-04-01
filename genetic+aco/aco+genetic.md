# Ant Colony Optimization and Genetic Algorithm.

ACO implementation to find shortest path in Maze, and Genetic algorithm implementation to solve a modified version of the Travelling Salesman Problem.

## Provided classes and Data
As part of the original assignment, mazes, coordinates, and optimal routes for TSP were provided.

Additionally, the Data classes in tsp_util and vizualization tools in viz_util were provided. Maze was a skeleton, and was mostly implenented from scratch. Other classes were also modified.

All classes in genetic and aco are implemeted from scratch.

## Ant Colony Optimization
Ant colony Optimization algorithm. Implemented ACO class, Ant class, Convergence class, and Ant helpers.

Used to calculate route length for modified TSP.

ACO can run with multiprocessing.

#### Additional Ant Features
1. Dynamic Alpha

    Problem Tackled: Premature convergence caused by an over-reliance on pheromones in the early stages leading to suboptimal solutions.
    Solution: Start with a lower alpha value, allowing more exploration at the beginning. Gradually increase alpha over iterations, focusing ants on exploiting the best-found paths later in the search process.

2. Alpha and Beta Sampling

    Problem Tackled: Lack of diversity in exploration strategies can lead to stagnation.
    Solution: Instead of all ants having the same alpha and beta values (which influences how much they focus on pheromone trails vs. heuristic factors), we sample alpha and beta values from a normal distribution.

3. Parameterized Elitism

    Problem Tackled: Fine-tuning the balance between preserving good solutions and promoting diverse exploration. Helps in finding tricky but faster routes 
    Solution: Introduce a parameter to control the number of "elite ants". The pheromone trails of these top-performing ants are given extra weight when updating pheromone levels after each iteration.

4. Ant Memory

    Problem Tackled: Ants wasting time revisiting areas and getting trapped in dead ends.
    Solution: Give each ant a simple memory to keep track of recently visited locations and avoid revisting them. In dead ends, use memory to backtrack efficiently.

## Genetic Algorithm
Genetic algorithm to solve modified TSP.

Implemented GeneticAlgorithm class, TSPData was provided and minimally modfied. 

Finds shortest route that visits all products in the maze. Can use pre-computed routes or use ACO class to find shortest paths between products (computationally intensive). Outputs path with optimal movements from start and when to pick up products.

Some issues with combination of ACO and GeneticAlgorithm. Does not correctly find best path. Fix is a work in progress.



