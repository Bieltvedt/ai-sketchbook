import numpy as np

class ACOConvergence:
    """
    Constructs ACOConvergence with params.
    Used to check for convergence in ACO

    @param max_iter : max iterations before stopping
    @param stagnation_threshold : max pheremone updates without imporovement before stopping.
    @param quality_threshold : maximum quality reached before stopping.
    @param consistency threshold : minimum max difference between pheremones in iteration i-1 and i before stopping

    NOTE : You can skip a convergence check by setting the corresponding variable to None.
    """
    def __init__(
            self, max_iter: int = 250, stagnation_threshold: int = 20, 
            quality_threshold: float = None, consistency_threshold: float = None
        ) -> None:
        self.max_iter = max_iter
        self.stagnation_threshold = stagnation_threshold
        self.quality_threshold = quality_threshold
        self.consistency_threshold = consistency_threshold
        
        self.iteration_count = 0
        self.best_quality = 1e9
        self.stagnation_count = 0
        self.last_pheromone_state = None

    """
    Adds iter iterations and checks if iterations < max iter
    @param iter: iterations since last call.
    """
    def check_iterations(self, iter: int = 1):
        # skip check if None
        if self.max_iter is None:
            return False
        
        self.iteration_count += iter
        if self.iteration_count >= self.max_iter:
            print("iteration convergence!")
        return self.iteration_count >= self.max_iter
    
    """
    Checks if improvement has stagnated for more than stagnation threshold iterations
    """
    def check_stagnation(self, current_quality: float):
        # skip check if None
        if self.stagnation_threshold is None:
            return False
       
        if current_quality < self.best_quality:
            self.best_quality = current_quality
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.stagnation_count >= self.stagnation_threshold:
            print("stagnation convergence!")
        return self.stagnation_count >= self.stagnation_threshold
    
    """
    Check if quality has met threshold
    """
    def check_quality(self, current_quality: float):
        # skip check if None
        if self.quality_threshold is None:
            return False
        
        if current_quality <= self.quality_threshold:
            print("quality convergence!")

        return current_quality <= self.quality_threshold
    
    """
    Check if pheremones have stopped changing
    """
    def check_consistency(self, pheremones: np.array):
        # skip check if None
        if self.consistency_threshold is None:
            return False
        
        if self.last_pheromone_state is not None:
            diff = np.abs(pheremones - self.last_pheromone_state).max()
            self.last_pheromone_state = pheremones
            if diff <= self.consistency_threshold:
                print("consistency convergence!")
            return diff <= self.consistency_threshold
        else:
            self.last_pheromone_state = pheremones
            return False
        
    """
    Checks if ACO has converged.
    Uses check_iterations(), check_stagnation(), check_quality(), and check_consistency()
    """
    def has_converged(self, current_quality: float, pheremones: np.array, iter: int = 1):
        return (
            self.check_iterations(iter) or
            self.check_stagnation(current_quality) or
            self.check_quality(current_quality) or 
            self.check_consistency(pheremones))
    
    """
    Resets ACO
    """
    def reset(self):
        self.iteration_count = 0
        self.best_quality = float('inf')
        self.stagnation_count = 0
        self.last_pheromone_state = None
