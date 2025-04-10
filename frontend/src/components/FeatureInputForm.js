import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, Paper, Grid, Container, Box, Divider, Chip, Alert, Avatar, MenuItem, Dialog, DialogTitle, DialogContent, DialogActions, CircularProgress} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { teal, deepOrange, indigo } from '@mui/material/colors';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import RoadIcon from '@mui/icons-material/Signpost';
import TrafficIcon from '@mui/icons-material/Traffic';
import WbSunnyIcon from '@mui/icons-material/WbSunny';
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
    TIME: '12:00',
    LATITUDE: '',
    LONGITUDE: '',
    ROAD_CLASS: 'Major Arterial',
    DISTRICT: 'Scarborough',
    ACCLOC: 'At Intersection',
    TRAFFCTL: 'Traffic Signal',
    VISIBILITY: 'Clear',
    LIGHT: 'Daylight',
    RDSFCOND: 'Dry'
  });

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
      <div className="predictor-container">
        <Container maxWidth="md">
          <Paper elevation={6} className="main-paper">
            {/* Header */}
            <div className="header-container">
              <Avatar className="header-avatar" sx={{ bgcolor: indigo[500] }}>
                <Typography variant="h4">!</Typography>
              </Avatar>
              <Typography variant="h4" className="header-title">Accident Risk Predictor</Typography>
              <Typography variant="subtitle1" className="header-subtitle">
                Complete all fields to predict accident severity
              </Typography>
            </div>

            {/* Form Section */}
            <Box component="form" onSubmit={handleSubmit} className="form-container">
              {/* Location */}
              <div className="form-section-container">
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
              <div className="form-section-container">
                <Typography variant="h6" className="section-title">
                  <LocationOnIcon fontSize="small"/>
                  Location
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
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
                  <Grid item xs={12} sm={4}>
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

              {/* Road Conditions */}
              <div className="form-section-container">
                <Typography variant="h6" className="section-title">
                  <RoadIcon fontSize="small"/>
                  Road Conditions
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                        select
                        fullWidth
                        label="Road Class"
                        name="ROAD_CLASS"
                        value={features.ROAD_CLASS}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['Major Arterial', 'Minor Arterial', 'Collector', 'Local', 'Expressway'].map(option => (
                          <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                        select
                        fullWidth
                        label="District"
                        name="DISTRICT"
                        value={features.DISTRICT}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['Scarborough', 'North York', 'Etobicoke York', 'Toronto and East York'].map(option => (
                          <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                        select
                        fullWidth
                        label="Road Surface"
                        name="RDSFCOND"
                        value={features.RDSFCOND}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['Dry', 'Wet', 'Snow', 'Ice', 'Slush'].map(option => (
                          <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {/* Traffic Conditions */}
              <div className="form-section-container">
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
                        label="Accident Location"
                        name="ACCLOC"
                        value={features.ACCLOC}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['At Intersection', 'Non Intersection', 'At/Near Private Drive', 'Intersection Related'].map(option => (
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

              {/* Weather Conditions */}
              <div className="form-section-container">
                <Typography variant="h6" className="section-title">
                  <WbSunnyIcon fontSize="small"/>
                  Weather Conditions
                </Typography>
                <Divider className="section-divider"/>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                        select
                        fullWidth
                        label="Visibility"
                        name="VISIBILITY"
                        value={features.VISIBILITY}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['Clear', 'Rain', 'Snow', 'Fog', 'Freezing Rain'].map(option => (
                          <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                        select
                        fullWidth
                        label="Lighting"
                        name="LIGHT"
                        value={features.LIGHT}
                        onChange={handleChange}
                        variant="outlined"
                        margin="normal"
                    >
                      {['Daylight', 'Dark', 'Dark, artificial', 'Dusk', 'Dawn'].map(option => (
                          <MenuItem key={option} value={option}>{option}</MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </div>

              {error && (
                  <Alert severity="error" sx={{mb: 2}}>
                    {error}
                  </Alert>
              )}

              <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={loading}
                  sx={{mt: 3, mb: 2, py: 1.5}}
              >
                {loading ? (
                    <>
                      <CircularProgress size={24} sx={{mr: 1}}/>
                      Analyzing...
                    </>
                ) : 'Predict Severity'}
              </Button>
            </Box>
          </Paper>
        </Container>

        {/* Results Dialog */}
        <Dialog
            open={openResults}
            onClose={handleCloseResults}
            maxWidth="sm"
            fullWidth
        >
          <DialogTitle>
            <Typography variant="h6" fontWeight="bold">Prediction Results</Typography>
          </DialogTitle>
          <DialogContent sx={{textAlign: 'center', py: 4}}>
            <Box
                sx={{
                  width: 180,
                  height: 180,
                  borderRadius: '50%',
                  display: 'inline-flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  border: `8px solid ${prediction === 1 ? deepOrange[500] : teal[500]}`,
                  backgroundColor: prediction === 1 ? 'rgba(255, 152, 0, 0.1)' : 'rgba(0, 150, 136, 0.1)',
                  color: prediction === 1 ? deepOrange[900] : teal[900],
                  mb: 3
                }}
            >
              <Typography variant="h3" fontWeight="bold">
                {Math.round(probability * 100)}%
              </Typography>
              <Typography variant="subtitle1" fontWeight="medium">
                {prediction === 1 ? 'HIGH RISK' : 'LOW RISK'}
              </Typography>
            </Box>
            <Typography variant="body1" paragraph>
              {prediction === 1
                  ? "High probability of severe outcomes"
                  : "Low probability of severe outcomes"}
            </Typography>
            <Chip
                label={`Confidence: ${(probability * 100).toFixed(1)}%`}
                color={prediction === 1 ? "warning" : "primary"}
                variant="outlined"
                sx={{fontSize: '0.9rem', px: 2}}
            />
          </DialogContent>
          <DialogActions sx={{justifyContent: 'center', pb: 3}}>
            <Button
              onClick={handleCloseResults}
              variant="contained"
              color="primary"
              sx={{ px: 4 }}
            >
              Done
            </Button>
          </DialogActions>
        </Dialog>
      </div>
    </ThemeProvider>
  );
};

export default FeatureInputForm;