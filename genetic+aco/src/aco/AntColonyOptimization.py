import numpy as np
import random
import os
import multiprocessing

from aco.AntHelpers import AntHelpers
from aco.IntelligentAnt import IntelligentAnt
from aco.ACOConvergence import ACOConvergence

from multiprocessing import Pool
from typing import Dict
from tsp_util.PathSpecification import PathSpecification
from tsp_util.Coordinate import Coordinate
from tsp_util.Route import Route
from tsp_util.Direction import Direction
from tsp_util.Maze import Maze

# Worker function for multiprocessing, ant_class should be ant class to use and params a dict of params to use
def run_ant_worker_int_test(maze: Maze, path_spec: PathSpecification, ant_class = IntelligentAnt, params = {}) -> Route:
    ant_worker = ant_class(maze=maze, path_specification=path_spec, **params)
    return ant_worker.find_route()

# Class representing the complete ACO algorithm.
# Finds shortest path between two points in a maze according to a path specification.
class AntColonyOptimization:
    """
    Constructs a new optimization object using the ant algorithm
    @param maze: the maze (environment) for ants
    @param ants_per_gen: the number of ants per generation (between update of pheromones)
    @param generations: the total number of generations of ants (pheromone updates)
    @param q: the normalization factor for the amount of dropped pheromone
    @param evaporation: the evaporation factor for the pheromones
    @param ant_class : Class of ants to spawn, defaults to StandardAnt
    @param convergence : ACOConvergence to use to check for convergence.
    """
    def __init__(
            self, maze: Maze, ants_per_gen: int, generations: int, 
            q: float, evaporation: float, elitism: float = 1, n_elite: int = 3,
            alpha: float = 1, beta: float = 1, alpha_increase: float = 0.01, alpha_beta_dev: float = 0,
            ant_class = IntelligentAnt, convergence: ACOConvergence = ACOConvergence()
        ) -> None:
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.elitism = elitism
        self.n_elite = n_elite
        self.alpha = alpha
        self.beta = beta
        self.alpha_beta_dev = alpha_beta_dev
        self.alpha_increase = alpha_increase
        self.ant_class = ant_class
        self.convergence = convergence

        self.iterations = 0

    """
    Loop that starts the shortest path process, updates pheremones once per generation

    @param path_specification: description of the route we wish to optimize
    @param ant_params : parameters to use to initialize ants (other than maze and path spec).
    @return the optimized route according to the ACO algorithm
    """
    def find_shortest_route(self, path_specification: PathSpecification) -> Route:
        # Setup params
        self.convergence.reset()
        self.iterations = 0
        best_route = None
        best_route_length = float('inf')
        cur_iter = 0
        # Loop through generations & ants
        for generation in range(self.generations):
            routes = []
            best_routes = []
            ant_params = {'alpha':self.alpha, 'beta':self.beta, 'alpha_beta_dev':self.alpha_beta_dev}
            self.iterations += 1
            cur_iter += 1
            for i in range(self.ants_per_gen):
                

                # Compute route
                route = run_ant_worker_int_test(self.maze, path_specification, self.ant_class, ant_params)
                routes.append(route) 

                # Check if best
                if len(best_routes) < self.n_elite or route.size() < best_routes[-1].size():
                    best_routes.append(route)
                    best_routes.sort(key=lambda x: x.size())
                    if len(best_routes) > self.n_elite:
                        best_routes.pop() 

            # Evaporate + add pheremones on routes
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(routes, self.q)

            # Elitism
            self.maze.add_pheromone_routes(best_routes, self.q * self.elitism)
            best_route = best_routes[0]

            # Apply alpha increase
            self._apply_alpha_increase()

            # Check for convergence
            if self.convergence.has_converged(
                current_quality=best_route.size(), pheremones=self.maze.get_pheremones(), 
                iter = self.iterations):
                self.iterations = cur_iter
                return best_route
        self.iterations = cur_iter
        return best_route
        
    """
    Finds shortest route using multiprocessing to run each ant in parrallel
    @param path_specification: description of the route we wish to optimize
    @param processors: #processors to use, defaults to system cpu count
    @param ant_params : parameters to use to initialize ants (other than maze and path spec).
    @return the optimized route according to the ACO algorithm
    """
    def find_shortest_route_multiproc(self, path_specification: PathSpecification, n_processors: int = os.cpu_count()
        ) -> Route:
        # Setup params
        self.convergence.reset()
        self.iterations = 0
        best_route = None
        best_route_length = float('inf')

        # Open pool with n_proccessors processors
        with Pool(n_processors) as pool:
            # Loop through generations
            for generation in range(self.generations):
                ant_params = {'alpha':self.alpha, 'beta':self.beta, 'alpha_beta_dev':self.alpha_beta_dev}
                self.iterations += self.ants_per_gen

                # Each ant gets *shared* maze, path spec, ant class, and ant params
                ant_tasks = [(self.maze, path_specification, self.ant_class, ant_params) for _ in range(self.ants_per_gen)]
                
                # run in parrallel
                routes = pool.starmap(run_ant_worker_int_test, ant_tasks)

                best_routes = []

                # Evaporate + Update pheremones
                self.maze.evaporate(self.evaporation)
                for route in routes:
                    if len(best_routes) < self.n_elite or route.size() < best_routes[-1].size():
                        best_routes.append(route)
                        best_routes.sort(key=lambda x: x.size())
                        if len(best_routes) > self.n_elite:
                            best_routes.pop() 


                self.maze.add_pheromone_routes(routes, self.q)

                # Elitism
                self.maze.add_pheromone_routes(best_routes, self.q * self.elitism)
                best_route = best_routes[0]
                # Apply alpha increase
                self._apply_alpha_increase()

                # Check for convergence after each generation
                x = 0
                if self.convergence.has_converged(current_quality=best_route_length, pheremones=self.maze.get_pheremones(), iter = self.ants_per_gen):
                    return best_route
       
       
            return best_route
            


    def _apply_alpha_increase(self):
        self.alpha = self.alpha * (1 + self.alpha_increase)