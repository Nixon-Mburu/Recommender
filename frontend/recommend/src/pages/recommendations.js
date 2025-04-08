import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import '../styles/recommendations.css';

const Recommendations = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const movies = location.state?.movies || [];
  const [filters, setFilters] = useState({
    genres: [],
    yearRange: [1900, 2025],
    minRating: 0
  });
  const [activeGenres, setActiveGenres] = useState([]);
  const [filteredMovies, setFilteredMovies] = useState(movies);
  const [isLoading, setIsLoading] = useState(true);

  // Extract all unique genres from movies
  useEffect(() => {
    if (movies.length) {
      const allGenres = [...new Set(movies.flatMap(movie => movie.genres))];
      setFilters(prev => ({ ...prev, genres: allGenres }));
      setIsLoading(false);
    }
  }, [movies]);

  // Apply filters
  useEffect(() => {
    const results = movies.filter(movie => {
      const yearMatch = movie.year >= filters.yearRange[0] && movie.year <= filters.yearRange[1];
      const ratingMatch = movie.predicted_rating >= filters.minRating;
      const genreMatch = activeGenres.length === 0 || 
        movie.genres.some(genre => activeGenres.includes(genre));
      
      return yearMatch && ratingMatch && genreMatch;
    });
    
    setFilteredMovies(results);
  }, [movies, filters.yearRange, filters.minRating, activeGenres]);

  const toggleGenre = (genre) => {
    setActiveGenres(prev => 
      prev.includes(genre) 
        ? prev.filter(g => g !== genre) 
        : [...prev, genre]
    );
  };

  if (isLoading) {
    return (
      <div className="recommendations-container">
        <div className="loading-container">
          <div className="pulse-loader">
            <div className="pulse"></div>
            <p className="pulse-text">Finding your perfect movies...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!movies.length) {
    return (
      <div className="recommendations-container">
        <div className="error-container">
          <div className="error-content">
            <div className="error-icon">😕</div>
            <h2>No Recommendations Found</h2>
            <p className="error-description">
              We couldn't find any movies matching your preferences. Try adjusting your criteria.
            </p>
            <button className="btn-primary" onClick={() => navigate('/')}>
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="recommendations-container">
      <div className="recommendations-header">
        <h1>Your Movie Recommendations</h1>
        <p className="subtitle">Personalized picks based on your preferences</p>
      </div>
      
      <div className="content-wrapper">
        <div className="filters-sidebar">
          <div className="filter-section">
            <h3>Genres</h3>
            <div className="genre-filters">
              {filters.genres.map(genre => (
                <label key={genre} className="filter-checkbox">
                  <input 
                    type="checkbox"
                    checked={activeGenres.includes(genre)}
                    onChange={() => toggleGenre(genre)}
                  />
                  {genre}
                </label>
              ))}
            </div>
          </div>
          
          <div className="filter-section">
            <h3>Year Range</h3>
            <div className="year-range-slider">
              <div className="slider-labels">
                <span>{filters.yearRange[0]}</span>
                <span>{filters.yearRange[1]}</span>
              </div>
              <input
                type="range"
                min="1900"
                max="2025"
                value={filters.yearRange[0]}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  yearRange: [parseInt(e.target.value), prev.yearRange[1]]
                }))}
              />
              <input
                type="range"
                min="1900"
                max="2025"
                value={filters.yearRange[1]}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  yearRange: [prev.yearRange[0], parseInt(e.target.value)]
                }))}
              />
            </div>
          </div>
          
          <div className="filter-section">
            <h3>Minimum Rating</h3>
            <div className="year-range-slider">
              <div className="slider-labels">
                <span>0</span>
                <span>5</span>
              </div>
              <input
                type="range"
                min="0"
                max="5"
                step="0.5"
                value={filters.minRating}
                onChange={(e) => setFilters(prev => ({
                  ...prev,
                  minRating: parseFloat(e.target.value)
                }))}
              />
              <div className="slider-value">
                {filters.minRating.toFixed(1)}+ ⭐
              </div>
            </div>
          </div>
          
          <button className="btn-primary" onClick={() => navigate('/')}>
            Back to Preferences
          </button>
        </div>
        
        <div className="results-grid">
          <h2>Found {filteredMovies.length} movies for you</h2>
          
          {filteredMovies.length === 0 ? (
            <div className="no-filtered-results">
              <p>No movies match your current filters. Try adjusting your criteria.</p>
            </div>
          ) : (
            <div className="movie-grid">
              {filteredMovies.map((movie) => (
                <div key={movie.id} className="movie-card">
                  <div className="movie-poster-container">
                    {movie.posterUrl ? (
                      <img
                        className="movie-poster"
                        src={movie.posterUrl}
                        alt={movie.title}
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.parentNode.innerHTML = `<div class="movie-poster-placeholder">🎬</div>`;
                        }}
                      />
                    ) : (
                      <div className="movie-poster-placeholder">🎬</div>
                    )}
                    
                    <div className="movie-rating">
                      <span className="rating-star">★</span>
                      {movie.predicted_rating.toFixed(1)}
                    </div>
                    
                    <div className="movie-overlay">
                      {movie.overview && (
                        <p className="movie-overview">{movie.overview}</p>
                      )}
                      {(movie.director || movie.actors) && (
                        <>
                          {movie.director && <p><strong>Director:</strong> {movie.director}</p>}
                          {movie.actors && <p><strong>Cast:</strong> {movie.actors}</p>}
                        </>
                      )}
                    </div>
                  </div>
                  
                  <div className="movie-info">
                    <h3 className="movie-title">{movie.title}</h3>
                    <p className="movie-year">{movie.year}</p>
                    
                    <div className="movie-genres">
                      {movie.genres.map((genre, index) => (
                        <span key={index} className="genre-tag">
                          {genre}
                        </span>
                      ))}
                    </div>
                    
                    {movie.imdbRating && (
                      <p>IMDb: {movie.imdbRating}/10</p>
                    )}
                    
                    <button className="btn-details">See Details</button>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          <div className="action-buttons">
            <button className="btn-primary">Load More Movies</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Recommendations;