import React, { useState } from 'react';
import axios from 'axios';
import {TextField, Button, Typography, Paper, Grid, Container, Box, Divider, Alert, MenuItem, CircularProgress} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import PersonIcon from '@mui/icons-material/Person';
import TrafficIcon from '@mui/icons-material/Traffic';
import ResultsDialog from '../ResultsDialog/ResultsDialog';
import './FeatureInputForm.css';
import theme from "../../styles/theme";

const FeatureInputForm = () => {
  const [features, setFeatures] = useState({
    TIME: '12:00',
    LATITUDE: '',
    LONGITUDE: '',
    IMPACTYPE: 'Pedestrian',
    VEHTYPE: 'Automobile',
    MANOEUVER: 'Going Straight',
    DRIVACT: 'Normal',
    INITDIR: 'Northbound',
    TRAFFCTL: 'Traffic Signal',
    INVAGE: '30-39',
    DRIVCOND: 'Normal',
    INVTYPE: 'Driver'
  });
  const [selectedModel, setSelectedModel] = useState("random_forest");
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openResults, setOpenResults] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFeatures(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await axios.post('http://localhost:5000/predict', {
        model: selectedModel,
        features: {
          ...features,
          LATITUDE: parseFloat(features.LATITUDE),
          LONGITUDE: parseFloat(features.LONGITUDE)
        }
      });
      setPrediction(res.data.prediction);
      setProbability(res.data.probability);
      setOpenResults(true);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCloseResults = () => {
    setOpenResults(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <div className="container">
        <Container maxWidth="md">
          <Paper elevation={3}>
            {/* Header */}
            <div className="header">
              <div className="header-content">
                <img
                  src="https://www.tps.ca/static/images/toronto-police-service-logo.png"
                  alt="Toronto Police Service"
                  className="logo"
                />
                <Typography variant="h4" className="title">Collision Predictor</Typography>
                <Typography variant="subtitle1">Complete all fields to predict accident severity</Typography>
              </div>
            </div>

            {/* Form Section */}
            <Box component="form" onSubmit={handleSubmit} className="form">
              {/* Time Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  <AccessTimeIcon fontSize="small" />
                  Time
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Time (HH:MM)"
                      name="TIME"
                      value={features.TIME}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                      required
                    />
                  </Grid>
                </Grid>
              </div>

              {/* Location Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  <LocationOnIcon fontSize="small"/>
                  Location
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Latitude"
                      name="LATITUDE"
                      value={features.LATITUDE}
                      onChange={handleChange}
                      type="number"
                      inputProps={{step: 0.0001}}
                      variant="outlined"
                      margin="normal"
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Longitude"
                      name="LONGITUDE"
                      value={features.LONGITUDE}
                      onChange={handleChange}
                      type="number"
                      inputProps={{step: 0.0001}}
                      variant="outlined"
                      margin="normal"
                      required
                    />
                  </Grid>
                </Grid>
              </div>

              {/* Accident Details Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  <DirectionsCarIcon fontSize="small"/>
                  Accident Details
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Impact Type"
                      name="IMPACTYPE"
                      value={features.IMPACTYPE}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Pedestrian', 'Turning Movement', 'Angle', 'Rear End', 'Sideswipe', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Vehicle Type"
                      name="VEHTYPE"
                      value={features.VEHTYPE}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Automobile', 'Motorcycle', 'Truck', 'Bicycle', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Manoeuvre"
                      name="MANOEUVER"
                      value={features.MANOEUVER}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Going Straight', 'Turning Left', 'Turning Right', 'Stopped', 'Reversing', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Driver Action"
                      name="DRIVACT"
                      value={features.DRIVACT}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Normal', 'Following Too Close', 'Improper Turn', 'Speeding', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {/* Traffic Conditions Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  <TrafficIcon fontSize="small"/>
                  Traffic Conditions
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Initial Direction"
                      name="INITDIR"
                      value={features.INITDIR}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Northbound', 'Southbound', 'Eastbound', 'Westbound', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Traffic Control"
                      name="TRAFFCTL"
                      value={features.TRAFFCTL}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Traffic Signal', 'Stop Sign', 'No Control', 'Pedestrian Crossover'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {/* Involved Party Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  <PersonIcon fontSize="small"/>
                  Involved Party
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Age Group"
                      name="INVAGE"
                      value={features.INVAGE}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['0-19', '20-29', '30-39', '40-49', '50-64', '65+'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      select
                      fullWidth
                      label="Driver Condition"
                      name="DRIVCOND"
                      value={features.DRIVCOND}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Normal', 'Impaired', 'Fatigued', 'Other'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid xs={12}>
                    <TextField
                      select
                      fullWidth
                      label="Involved Type"
                      name="INVTYPE"
                      value={features.INVTYPE}
                      onChange={handleChange}
                      variant="outlined"
                      margin="normal"
                    >
                      {['Driver', 'Passenger', 'Pedestrian', 'Cyclist'].map(option => (
                        <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {/* Model Selection Section */}
              <div className="form-section">
                <Typography variant="h6" className="section-title">
                  Model Selection
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      select
                      fullWidth
                      label="Select Prediction Model"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      variant="outlined"
                      margin="normal"
                    >
                      <MenuItem value="random_forest">Random Forest</MenuItem>
                      <MenuItem value="xgboost">SVM</MenuItem>
                      <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {error}
                </Alert>
              )}

              <Box textAlign="center" mt={4} mb={3}>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading}
                  className="submit-button"
                >
                  {loading ? (
                    <>
                      <CircularProgress size={24} sx={{ mr: 1, color: 'white' }}/>
                      Analyzing...
                    </>
                  ) : 'Predict Severity'}
                </Button>
              </Box>
            </Box>
          </Paper>
        </Container>

        {/* Results Dialog */}
        <ResultsDialog
          open={openResults}
          onClose={handleCloseResults}
          probability={probability}
          prediction={prediction}
          selectedModel={selectedModel}
        />
      </div>
    </ThemeProvider>
  );
};

export default FeatureInputForm;