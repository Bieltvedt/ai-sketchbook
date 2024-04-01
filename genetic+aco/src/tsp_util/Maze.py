import numpy as np

from typing import List
from tsp_util.Coordinate import Coordinate
from tsp_util.Route import Route
from tsp_util.Direction import Direction
from tsp_util.SurroundingPheromone import SurroundingPheromone

class Maze:

    """
    Constructor of a Maze
    @param walls: array of ints representing the accessible (1) and inaccessible (0) tiles
    @param width: the width (horizontal dimension) of the Maze
    @param length: the length (vertical dimension) of the Maze

    NOTE : No nested list support so no type hints for walls
    NOTE : pheremones and walls have shared ar ray versions for multiproc.
    """
    def __init__(self, walls, width: int, length: int) -> None:
        self.walls = np.array(walls, dtype=int).T
        # self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.pheremones = None

        self.initialize_pheromones()

    """
    Initialize pheromones on all tiles of the Maze
    """
    def initialize_pheromones(self, init_pheremone: float = 1) -> None:
        self.pheremones = np.zeros((self.length, self.width))
        # Use walls as mask to update pheremones
        self.pheremones += self.walls * init_pheremone

    """
    Reset the Maze for a new shortest path problem
    """
    def reset(self) -> None:
        self.initialize_pheromones()
 
    """
    Update the pheromones along a certain route according to a certain Q
    Ignores routes that contain only start
    @param route: the route taken by an ant
    @param q: the normalization factor for the amount of dropped pheromone

    NOTE : Using delta_tk = Q * 1 / L (from lecture slides)
    """
    def add_pheromone_route(self, route: Route, q: float) -> None:
        # Skip if route contains only start
        if route.size() == 0:
            return
        
        # Find delta tk
        delta_tk = q * 1 / (route.size() + 0.000001)

        # add to start
        cur = route.get_start()
        assert self.in_bounds(cur) and self.walls[cur.get_tuple()] > 0  # assert update is valid
        self.pheremones[cur.get_tuple()] += delta_tk

        # add to rest of route
        for dir in route.get_route():
            cur = cur.add_direction(dir)
            assert self.in_bounds(cur) and self.walls[cur.get_tuple()] > 0  # assert update is valid
            self.pheremones[cur.get_tuple()] += delta_tk              

    """
    Update pheromones for a list of routes
    @param routes: a list of routes taken by the ants
    @param q: the normalization factor for the amount of dropped pheromone
    """
    def add_pheromone_routes(self, routes: List[Route], q: float) -> None:
        for r in routes:
            self.add_pheromone_route(r, q)

    """
    Evaporate pheromone
    @param rho: the evaporation factor

    NOTE : Remember to call before adding pheremones to routes!
    """
    def evaporate(self, rho: float) -> None:
        self.pheremones *= (1 - rho)

    """
    Getter for the width of the maze
    @return the width of the maze
    """
    def get_width(self) -> int:
        return self.width

    """
    Getter for the length of the maze
    @return the length of the maze
    """
    def get_length(self) -> int:
        return self.length

    """
    Returns a the amount of pheromones on the neighbouring positions (N/S/E/W)
    @param position: the coordinate where we need to check the surrounding pheromones
    @return the pheromones on the neighbouring coordinates.
    """
    def get_surrounding_pheromone(self, position: Coordinate) -> SurroundingPheromone:
        # Find surrounding pheromones.
        north = self.get_pheromone(position.add_direction(Direction.north))
        east = self.get_pheromone(position.add_direction(Direction.east))
        south = self.get_pheromone(position.add_direction(Direction.south))
        west = self.get_pheromone(position.add_direction(Direction.west))

        return SurroundingPheromone(north, east, south, west)

    """
    Getter for the pheromones on a specific coordinate.
    If the position is not in bounds returns 0
    @param pos: coordinate for the poition of interest
    @return the amount of pheromone at the specified poition
    """
    def get_pheromone(self, pos: Coordinate) -> float:
        # Return 0 if out of bounds
        if not self.in_bounds(pos):
            return 0
        return self.pheremones[pos.get_tuple()]
    
    """
    Get pheremone array
    """
    def get_pheremones(self) -> np.array:
        return self.pheremones

    """
    Check whether a coordinate lies in the bounds of the current maze
    @param position: the position that we need to check
    @return true if the coordinate lies within the current maze
    """
    def in_bounds(self, position: Coordinate) -> bool:
        return position.x_between(0, self.width) and position.y_between(0, self.length)
    
    """
    Get list of valid directions you can travel from a point
    """
    def get_valid_dir(self, pos: Coordinate) -> List[Direction]:
        # Setup
        valid_dir = []

        # Loop through all directions
        for dir in [Direction.north, Direction.east, Direction.south, Direction.west]:
            new_pos = pos.add_direction(dir)
            
            # Add if valid
            if self.in_bounds(new_pos) and self.pheremones[new_pos.get_tuple()] > 0:
                    valid_dir.append(dir)
        return valid_dir

    def max_manhattan_dist(self):
        return self.width + self.length
    
    """
    Representation of Maze as defined by the input file format.
    @return the human-readable representation of a maze
    """
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    """
    Method that builds a maze from a file
    @param file_path: path to the file which stores the maze
    @return a maze object with pheromones initialized to 0s on inaccessible and 1s on accessible tiles
    """
    @staticmethod
    def create_maze(file_path: str):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)