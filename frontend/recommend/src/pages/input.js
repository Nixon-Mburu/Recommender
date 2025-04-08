// src/pages/input.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import "../styles/input.css";

const Input = () => {
  const navigate = useNavigate();
  const [preferences, setPreferences] = useState({
    preferredGenres: [],
    yearRange: { from: 2000, to: 2023 },
    mood: '',
    age: '',
    gender: '',
    occupation: '',
    minimumRating: 7.0
  });

  const genres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller'
  ];

  const moods = [
    'Happy', 'Relaxed', 'Excited', 'Thoughtful', 
    'Romantic', 'Mysterious', 'Intense'
  ];

  const occupations = [
    'Student', 'Professional', 'Artist', 'Engineer',
    'Healthcare', 'Business', 'Education', 'Other'
  ];

  const handleGenreChange = (genre) => {
    setPreferences(prev => ({
      ...prev,
      preferredGenres: prev.preferredGenres.includes(genre)
        ? prev.preferredGenres.filter(g => g !== genre)
        : [...prev.preferredGenres, genre]
    }));
  };

  const handleMoodSelect = (mood) => {
    setPreferences(prev => ({
      ...prev,
      mood: prev.mood === mood ? '' : mood
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (preferences.preferredGenres.length === 0) {
      alert('Please select at least one genre');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preferences)
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || `Error: ${response.status}`);
      }
      
      if (data.recommendations && data.recommendations.length > 0) {
        navigate('/recommendations', { state: { movies: data.recommendations } });
      } else {
        alert('No movies found matching your preferences. Please try different criteria.');
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      alert(`Unable to get recommendations: ${error.message}. Please try again or adjust your preferences.`);
    }
  };

  return (
    <div className="input-container">
      <div className="content-wrapper">
        <div className="header-section">
          <h1>Movie Preferences</h1>
          <p className="subtitle">Tell us what you like, and we'll find the perfect movies for you</p>
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Select Your Favorite Genres</label>
            <div className="genres-grid">
              {genres.map(genre => (
                <div className="genre-item" key={genre}>
                  <input
                    type="checkbox"
                    id={genre}
                    checked={preferences.preferredGenres.includes(genre)}
                    onChange={() => handleGenreChange(genre)}
                  />
                  <label htmlFor={genre}>{genre}</label>
                </div>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label>Release Year Range</label>
            <div className="year-inputs">
              <input
                type="number"
                min="1900"
                max="2023"
                value={preferences.yearRange.from}
                onChange={e => setPreferences(prev => ({
                  ...prev,
                  yearRange: { ...prev.yearRange, from: parseInt(e.target.value) }
                }))}
              />
              <span>to</span>
              <input
                type="number"
                min="1900"
                max="2023"
                value={preferences.yearRange.to}
                onChange={e => setPreferences(prev => ({
                  ...prev,
                  yearRange: { ...prev.yearRange, to: parseInt(e.target.value) }
                }))}
              />
            </div>
          </div>

          <div className="form-group">
            <label>How old are you?</label>
            <input
              type="number"
              className="age-input"
              value={preferences.age}
              onChange={e => setPreferences(prev => ({
                ...prev,
                age: e.target.value
              }))}
              min="13"
              max="100"
              placeholder="Enter your age"
            />
          </div>

          <div className="form-group">
            <label>Gender</label>
            <div className="gender-selector">
              <button
                type="button"
                className={`gender-btn ${preferences.gender === 'M' ? 'active' : ''}`}
                onClick={() => setPreferences(prev => ({ ...prev, gender: 'M' }))}
              >
                Male
              </button>
              <button
                type="button"
                className={`gender-btn ${preferences.gender === 'F' ? 'active' : ''}`}
                onClick={() => setPreferences(prev => ({ ...prev, gender: 'F' }))}
              >
                Female
              </button>
              <button
                type="button"
                className={`gender-btn ${preferences.gender === 'O' ? 'active' : ''}`}
                onClick={() => setPreferences(prev => ({ ...prev, gender: 'O' }))}
              >
                Other
              </button>
            </div>
          </div>

          <div className="form-group">
            <label>What's your current mood?</label>
            <div className="mood-selector">
              {moods.map(mood => (
                <button
                  key={mood}
                  type="button"
                  className={`mood-btn ${preferences.mood === mood ? 'active' : ''}`}
                  onClick={() => handleMoodSelect(mood)}
                >
                  {mood}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label>Occupation</label>
            <select
              className="occupation-select"
              value={preferences.occupation}
              onChange={e => setPreferences(prev => ({
                ...prev,
                occupation: e.target.value
              }))}
            >
              <option value="">Select occupation</option>
              {occupations.map(occ => (
                <option key={occ} value={occ.toLowerCase()}>{occ}</option>
              ))}
            </select>
          </div>

          <button className="submit-btn" type="submit">
            Find Movies
          </button>
        </form>
      </div>
    </div>
  );
};

export default Input;
