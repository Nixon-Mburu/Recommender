import numpy as np

class MoviePreprocessor:
    def __init__(self, genres):
        self.genres = genres
        
    def transform(self, data):
        # Create genre features
        genre_features = np.zeros(len(self.genres))
        for genre in data.get('genres', []):
            if genre in self.genres:
                genre_features[self.genres.index(genre)] = 1
                
        # Create year features
        year_range = data.get('year_range', {'from': 2000, 'to': 2023})
        year_from = (year_range['from'] - 2000) / 23  # Normalize to [0,1]
        year_to = (year_range['to'] - 2000) / 23  # Normalize to [0,1]
        
        # Combine features
        features = np.concatenate([genre_features, [year_from, year_to]])
        return features
