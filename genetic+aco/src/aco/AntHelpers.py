from tsp_util.Coordinate import Coordinate

class AntHelpers:

    @staticmethod
    def manhattan_dist(start: Coordinate, end: Coordinate) -> float:
        # + 1 to prevent div by 0, min manhattan_dist = 1
        return 1 / (abs(start.get_x() - end.get_x()) + abs(start.get_y() - end.get_y()) + 1)
    
    @staticmethod
    def manhattan_dist_recip(start: Coordinate, end: Coordinate) -> float:
        return 1 / AntHelpers.manhattan_dist(start, end)

    @staticmethod
    def manhattan_dist_exp_decay(start: Coordinate, end: Coordinate) -> float:
        dist = AntHelpers.manhattan_dist(start, end)
        return np.exp(-dist)