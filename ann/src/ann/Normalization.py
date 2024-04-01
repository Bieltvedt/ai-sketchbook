import numpy as np


class Normalization:
    @staticmethod
    def min_max_scale(data):
        df = data.copy()
        mins = np.min(df, axis=0)  
        maxs = np.max(df, axis=0)  
        denom = maxs - mins  
        denom[denom == 0] = 1  # Handle cases where max == min (set to 1 to avoid NaN)
        return (df - mins) / denom
    
    @staticmethod
    def z_score(data):
        df = data.copy()
        eps = 1e-8  
        means = np.mean(df, axis=0)
        stds = np.std(df, axis=0)
        stds[stds == 0] = eps  # Handle cases where standard deviation is 0
        return (df - means) / stds
    
    @staticmethod
    def mean_shift(data):
        df = data.copy()
        scaled_df = df - df.mean()
        return scaled_df