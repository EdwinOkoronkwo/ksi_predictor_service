import React, { useState } from 'react';
import axios from 'axios';
import {TextField, Checkbox, FormControlLabel, Button, Typography, Paper, Grid, Slider, Container, Box, Card, Divider, LinearProgress, Chip, Alert, Avatar
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { teal, deepOrange, indigo } from '@mui/material/colors';
import './FeatureInputForm.css';

const theme = createTheme({
  palette: {
    primary: { main: indigo[700] },
    secondary: { main: teal[600] },
    warning: { main: deepOrange[500] },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h2: { fontWeight: 700, letterSpacing: 0.5 },
    h6: { fontWeight: 600 },
  },
});

const FeatureInputForm = () => {
  const [features, setFeatures] = useState({
    "categorical__IMPACTYPE_Pedestrian Collisions": 0,
    "numerical__LATITUDE": '',
    "categorical__VEHTYPE_Automobile, Station Wagon": 0,
    "numerical__LONGITUDE": '',
    "categorical__INITDIR_East": 0,
    "numerical__combined_xy": 0.5,
    "categorical__INITDIR_West": 0,
    "categorical__MANOEUVER_Going Ahead": 0,
    "categorical__ROAD_CLASS_Major Arterial": 0,
    "categorical__DRIVACT_Driving Properly": 0
  });

  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    setFeatures((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value,
    }));
  };

  const handleSliderChange = (name, value) => {
    setFeatures((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const submissionFeatures = {
        ...features,
        "numerical__LATITUDE": features["numerical__LATITUDE"] === '' ? null : parseFloat(features["numerical__LATITUDE"]),
        "numerical__LONGITUDE": features["numerical__LONGITUDE"] === '' ? null : parseFloat(features["numerical__LONGITUDE"])
      };

      const res = await axios.post('http://localhost:5000/predict', { features: submissionFeatures });
      setPrediction(res.data.prediction);
      setProbability(res.data.probability);
    }
    catch (err) {
      console.error(err);
      setError('Prediction failed. Please check your inputs and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <div className="predictor-container">
        <Container maxWidth="lg">
          <Paper elevation={6} className="main-paper">
            {/* Header */}
            <div className="header-container">
              <Avatar className="header-avatar">
                <Typography variant="h4">!</Typography>
              </Avatar>
              <Typography variant="h3" className="header-title">Accident Risk Predictor</Typography>
              <Typography variant="subtitle1" className="header-subtitle">
                Input road, vehicle, and behavioral data to predict severity
              </Typography>
            </div>

            {/* Main Content */}
            <div className="content-container">
              <Grid container spacing={0} className="main-grid">
                {/* Form Section */}
                <Grid item xs={12} md={6} className="form-section">
                  <Box component="form" onSubmit={handleSubmit} className="form-container">
                    {/* Location Section */}
                    <div className="form-section-container">
                      <Typography variant="h5" className="section-title">
                        <span>📌</span> Location Data
                      </Typography>
                      <Divider className="section-divider" />
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Latitude"
                            name="numerical__LATITUDE"
                            value={features["numerical__LATITUDE"]}
                            onChange={handleChange}
                            type="number"
                            inputProps={{ step: 0.0001 }}
                            variant="outlined"
                            className="form-field"
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Longitude"
                            name="numerical__LONGITUDE"
                            value={features["numerical__LONGITUDE"]}
                            onChange={handleChange}
                            type="number"
                            inputProps={{ step: 0.0001 }}
                            variant="outlined"
                            className="form-field"
                          />
                        </Grid>
                      </Grid>
                    </div>

                    {/* Collision Details */}
                    <div className="form-section-container">
                      <Typography variant="h5" className="section-title">
                        <span>🚗</span> Collision Details
                      </Typography>
                      <Divider className="section-divider" />
                      <div className="checkbox-group">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={features["categorical__IMPACTYPE_Pedestrian Collisions"] === 1}
                              onChange={handleChange}
                              name="categorical__IMPACTYPE_Pedestrian Collisions"
                              color="primary"
                            />
                          }
                          label="Pedestrian Collision"
                          className="form-label"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={features["categorical__VEHTYPE_Automobile, Station Wagon"] === 1}
                              onChange={handleChange}
                              name="categorical__VEHTYPE_Automobile, Station Wagon"
                              color="primary"
                            />
                          }
                          label="Automobile / Station Wagon"
                          className="form-label"
                        />
                      </div>
                    </div>

                    {/* Direction */}
                    <div className="form-section-container">
                      <Typography variant="h5" className="section-title">
                        <span>🧭</span> Direction
                      </Typography>
                      <Divider className="section-divider" />
                      <div className="checkbox-group">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={features["categorical__INITDIR_East"] === 1}
                              onChange={handleChange}
                              name="categorical__INITDIR_East"
                              color="primary"
                            />
                          }
                          label="Traveling East"
                          className="form-label"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={features["categorical__INITDIR_West"] === 1}
                              onChange={handleChange}
                              name="categorical__INITDIR_West"
                              color="primary"
                            />
                          }
                          label="Traveling West"
                          className="form-label"
                        />
                      </div>
                    </div>

                    {/* Road Conditions */}
                    <div className="form-section-container">
                      <Typography variant="h5" className="section-title">
                        <span>🛣️</span> Road Conditions
                      </Typography>
                      <Divider className="section-divider" />
                      <div className="checkbox-group">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={features["categorical__ROAD_CLASS_Major Arterial"] === 1}
                              onChange={handleChange}
                              name="categorical__ROAD_CLASS_Major Arterial"
                              color="primary"
                            />
                          }
                          label="Major Arterial Road"
                          className="form-label"
                        />
                      </div>
                      <div className="slider-container">
                        <Typography className="slider-label">Combined XY Value</Typography>
                        <Slider
                          min={0}
                          max={1}
                          step={0.01}
                          value={features["numerical__combined_xy"]}
                          onChange={(e, val) => handleSliderChange("numerical__combined_xy", val)}
                          valueLabelDisplay="auto"
                          className="form-slider"
                        />
                      </div>
                    </div>

                    <Button
                      type="submit"
                      fullWidth
                      variant="contained"
                      color="primary"
                      size="large"
                      disabled={loading}
                      className="submit-button"
                    >
                      {loading ? 'Analyzing...' : 'Predict Severity'}
                    </Button>
                  </Box>
                </Grid>

                {/* Results Section */}
                <Grid item xs={12} md={6} className="results-section">
                  <div className="results-container">
                    <Typography variant="h5" className="results-title">
                      <span>📊</span> Prediction Results
                    </Typography>
                    <Divider className="results-divider" />

                    {loading && (
                      <div className="loading-container">
                        <LinearProgress className="loading-progress" />
                        <Typography className="loading-text">Processing...</Typography>
                      </div>
                    )}

                    {error && (
                      <Alert severity="error" className="error-alert">
                        {error}
                      </Alert>
                    )}

                    {prediction !== null ? (
                      <Card className={`prediction-card ${prediction === 1 ? 'severe' : 'low'}`}>
                        <div className="risk-meter">
                          <Typography variant="h2" className="risk-percentage">
                            {Math.round(probability * 100)}%
                          </Typography>
                        </div>
                        <Typography variant="h4" className="risk-level">
                          {prediction === 1 ? 'Severe Risk' : 'Low Risk'}
                        </Typography>
                        <Typography className="risk-description">
                          {prediction === 1
                            ? "High probability of severe injuries or fatalities"
                            : "Low probability of severe outcomes"}
                        </Typography>
                        <Chip
                          label={`Confidence: ${(probability * 100).toFixed(1)}%`}
                          className="confidence-chip"
                          variant="outlined"
                        />
                      </Card>
                    ) : (
                      <div className="empty-results">
                        <Typography className="empty-results-text">
                          Submit the form to see prediction results
                        </Typography>
                      </div>
                    )}
                  </div>
                </Grid>
              </Grid>
            </div>
          </Paper>
        </Container>
      </div>
    </ThemeProvider>
  );
};

export default FeatureInputForm;