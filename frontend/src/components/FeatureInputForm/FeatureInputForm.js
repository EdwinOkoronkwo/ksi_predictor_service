import React, { useEffect, useState } from 'react';
import {TextField, Button, Typography, Paper, Grid, Container, Box, Divider, Alert, MenuItem, CircularProgress, ButtonGroup} from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import PersonIcon from '@mui/icons-material/Person';
import TrafficIcon from '@mui/icons-material/Traffic';
import ResultsDialog from '../ResultsDialog/ResultsDialog';
import './FeatureInputForm.css';
import {
  fetchAvailableModels,
  predictWithBestModel,
  predictWithSelectedModel
} from '../../api/modelApi';

const FeatureInputForm = () => {
  const [features, setFeatures] = useState({
    TIME: '09:00',
    LATITUDE: '43.700667',
    LONGITUDE: '-79.427836',
    IMPACTYPE: 'Pedestrian Collisions',
    VEHTYPE: 'Passenger Van',
    MANOEUVER: 'Going Ahead',
    DRIVACT: 'Lost control',
    INITDIR: 'East',
    TRAFFCTL: 'Traffic Signal',
    INVAGE: '40 to 49',
    DRIVCOND: 'Had been Drinking',
    INVTYPE: 'Passenger'
  });

  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [predictionMode, setPredictionMode] = useState('best');
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openResults, setOpenResults] = useState(false);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const models = await fetchAvailableModels();
        setAvailableModels(models);
        if (models.length > 0) {
          setSelectedModel(models[0]);
        }
      } catch (err) {
        setError('Failed to load available models');
      }
    };
    loadModels();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFeatures(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const formattedFeatures = {
        ...features,
        LATITUDE: parseFloat(features.LATITUDE),
        LONGITUDE: parseFloat(features.LONGITUDE)
      };

      let result;
      if (predictionMode === 'best') {
        result = await predictWithBestModel(formattedFeatures);
      } else {
        result = await predictWithSelectedModel(selectedModel, formattedFeatures);
      }

      setPrediction(result.prediction);
      setProbability(result.probability);
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
            {/* Prediction Mode Selection */}
            <Box sx={{ mb: 3 }}>
              <ButtonGroup fullWidth>
                <Button
                  variant={predictionMode === 'best' ? 'contained' : 'outlined'}
                  onClick={() => setPredictionMode('best')}
                >
                  Use Best Model
                </Button>
                <Button
                  variant={predictionMode === 'select' ? 'contained' : 'outlined'}
                  onClick={() => setPredictionMode('select')}
                >
                  Select Model
                </Button>
              </ButtonGroup>
            </Box>

            {/* Model Selection (only shown when in select mode) */}
            {predictionMode === 'select' && (
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
                      {availableModels.map((model) => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>
            )}

            {/* Time Section */}
            <div className="form-section">
              <Typography variant="h6" className="section-title">
                <AccessTimeIcon fontSize="small" />
                Time
              </Typography>
              <Divider className="section-divider"/>
              <Grid container spacing={2}>
                <Grid item xs={12}>
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
                    {['Pedestrian Collisions', 'Turning Movement', 'Angle', 'Rear End', 'Sideswipe', 'Other'].map(option => (
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
                    {['Passenger Van', 'Automobile', 'Motorcycle', 'Truck', 'Bicycle', 'Other'].map(option => (
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
                    {['Going Ahead', 'Going Straight', 'Turning Left', 'Turning Right', 'Stopped', 'Reversing', 'Other'].map(option => (
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
                    {['Lost control', 'Normal', 'Following Too Close', 'Improper Turn', 'Speeding', 'Other'].map(option => (
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
                    {['East', 'Northbound', 'Southbound', 'Westbound', 'Other'].map(option => (
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
                    {['40 to 49', '0-19', '20-29', '30-39', '50-64', '65+'].map(option => (
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
                    {['Had been Drinking', 'Normal', 'Impaired', 'Fatigued', 'Other'].map(option => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12}>
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
                    {['Passenger', 'Driver', 'Pedestrian', 'Cyclist'].map(option => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
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

      <ResultsDialog
        open={openResults}
        onClose={handleCloseResults}
        probability={probability}
        prediction={prediction}
        selectedModel={predictionMode === 'best' ? 'Best Model' : selectedModel}
      />
    </div>
  );
};

export default FeatureInputForm;