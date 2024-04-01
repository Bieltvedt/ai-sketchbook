import numpy as np
import random

from typing import Dict
from aco.AntHelpers import AntHelpers
from tsp_util.PathSpecification import PathSpecification
from tsp_util.Coordinate import Coordinate
from tsp_util.Route import Route
from tsp_util.Direction import Direction
from tsp_util.Maze import Maze


# Class that represents the basic Ant functionality
class IntelligentAnt:

    """
    Constructor of a StandardAnt taking a Maze and PathSpecification
    @param maze: the Maze where the ant will try to find a route
    @param path_specification: the PathSpecification consisting of a start and an end coordinate
    @param alpha : controlls the strength of t_ij in prob. calc. Defaults to 1
    @param beta : controlls the strength of n_ij (heuristic, manhattan dist here) in prob. calc. Defaults to 0.5
    @param max_steps : max steps ant can take before termination. Defaults to 500
    """
    def __init__(
            self, maze: Maze, path_specification: PathSpecification, 
            alpha: float = 1, beta: float = 1, alpha_beta_dev: float = 0
        ) -> None:
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
    
        self.alpha = np.random.normal(alpha, alpha_beta_dev)
        self.beta = np.random.normal(beta, alpha_beta_dev)

        self.rand = random
        self.steps = 0
        self.stop = False
        self.visited = []
        self.stuck = []
        

    """
    Method that performs a single complete run through the maze by the ant.
    @return the route found by the ant if Route only contains start -> no path found
    """
    def find_route(self) -> Route:
        # Setup
        self.route = Route(self.start)
        self.current_position = self.start
        self.steps = 0
        self.visited = [self.start]

        # Action loop
        while not self.stop:
            # Choose direction
            dir = self._choose_next()
            # Check if _choose_next() got stuck (returns None) -> return Route with only start
            if dir is None: 
                return Route(self.start)

            # Move + update visited & route
            self.current_position = self.current_position.add_direction(dir)
            
            # Update visited and route
            self.visited.append(self.current_position)
            self.route.add(dir)

            # Check stopping conditions, update alpha & beta
            self.steps+=1
            self._stop_searching()
        return self.route

    """
    Chooses next direction. Uses "roulette wheel selection" with self.rand.choice
    @return Direction to go next or None if none valid
    """
    def _choose_next(self) -> Direction:
        # Get probabilities of each dir
        probability_dict = self._get_probabilities()

        # No valid path, backtrack!
        if not probability_dict:
            # Able to backtrack
            if len(self.visited) > 1 and self.route.get_route():
                self.backtracking = True
                self.route.remove_last() # remove from route
                self.stuck.append(self.visited.pop())  # pop current val
                self.current_position = self.visited[-1] # set to last visited before current
                return self._choose_next()
            # Not able to backtrack, no path found! return None and dont use for update later
            else:
                return None
            
        # Choose next direction 
        directions = list(probability_dict.keys())
        probs = list(probability_dict.values())
        next_dir = self.rand.choices(directions, weights=probs, k=1)[0]
        return next_dir
    
    """
    Gets probabilities of going in each legal direction based on pheremones
    Uses manhattan distance to end as heuristic. (was 1/Lij on slides)
    @return Dictionary of directions (keys) and probabilities (values)
    """
    def _get_probabilities(self) -> Dict[Direction, float]:
        # Setup. Get valid directions and surrounding pheremones
        probabilities = {}
        valid_dir = self.maze.get_valid_dir(self.current_position)
        surrounding_pheromones = self.maze.get_surrounding_pheromone(self.current_position)

        # Loop through possible moves
        for dir in valid_dir:
            pos = self.current_position.add_direction(dir)
            
            if pos not in self.visited and pos not in self.stuck:
                # Scale by q to ensure similar scales
                eta = (self._get_heuristic(pos)) ** self.beta
                tau = surrounding_pheromones.get(dir) ** self.alpha
                probabilities[dir] = eta * tau
            
            assert self.maze.in_bounds(pos) and self.maze.walls[pos.get_tuple()] > 0    # Assert update is valid

        # Get final probabilities by dividing by sum
        total = sum(probabilities.values())
        if total == 0:
            return None
    
        probabilities = {key: value / total for key, value in probabilities.items()}

        return probabilities
    
    def _get_heuristic(self, pos: Coordinate) -> float:
        dist = AntHelpers.manhattan_dist(pos, self.end)
        max_dist = self.maze.max_manhattan_dist()
        # Ensure the heuristic value is within a useful range
        scaled_heuristic = (max_dist - dist) / (max_dist + 0.00000001)
        return scaled_heuristic
    
    """
    Checks if the ant is finished searching

    Currently stops if current = end or iter > max_iter
    """
    def _stop_searching(self) -> None:
        self.stop = self.current_position == self.end