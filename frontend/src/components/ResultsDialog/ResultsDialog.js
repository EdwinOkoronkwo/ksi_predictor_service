import React from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Typography, Paper, Button, Box, LinearProgress
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import { useTheme } from '@mui/material/styles';
import './ResultsDialog.css';

const ResultsDialog = ({ open, onClose, probability, prediction, selectedModel }) => {
  const theme = useTheme();

  const severityConfig = {
    fatal: {
      color: theme.palette.error.main,
      bgColor: theme.palette.error.light,
      icon: <WarningIcon sx={{ color: theme.palette.error.main, fontSize: 48 }} />,
      text: 'FATAL'
    },
    nonfatal: {
      color: theme.palette.primary.main,
      bgColor: theme.palette.primary.light,
      icon: <CheckCircleIcon sx={{ color: theme.palette.primary.main, fontSize: 48 }} />,
      text: 'NON-FATAL'
    }
  };

  const severity = prediction === 1 ? severityConfig.fatal : severityConfig.nonfatal;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <div className="severity-indicator" style={{ backgroundColor: severity.color }}/>
      <DialogTitle>
        <div className="dialog-header">
          <img
            src="https://www.tps.ca/static/images/toronto-police-service-logo.png"
            alt="Toronto Police Service"
            className="logo"
          />
          <Typography variant="h5">Results</Typography>
        </div>
        <Typography variant="body2">
          Prediction Model: {selectedModel.replace('_', ' ').toUpperCase()}
        </Typography>
        <Typography variant="body2">
          Prediction: {prediction}
        </Typography>
      </DialogTitle>

      <DialogContent>
        <Paper
          elevation={0}
          className="severity-card"
          style={{ backgroundColor: severity.bgColor }}
        >
          <div className="severity-content">
            {severity.icon}
            <Typography
              variant="h5"
              style={{
                fontWeight: theme.typography.fontWeightBold,
                color: severity.color
              }}
            >
              {severity.text}
            </Typography>
          </div>

          <Box className="probability-container">
            <Typography variant="subtitle1" className="probability-label">
              Probability:
            </Typography>
            <LinearProgress
              variant="determinate"
              value={probability * 100}
              className="probability-bar"
              sx={{
                '& .MuiLinearProgress-bar': {
                  backgroundColor: probability > 0.7
                    ? theme.palette.error.main
                    : theme.palette.primary.main
                }
              }}
            />
            <Typography variant="body1" className="probability-value">
              {(probability * 100).toFixed(1)}%
            </Typography>
          </Box>
        </Paper>
      </DialogContent>

      <DialogActions>
        <Button
          onClick={onClose}
          variant="contained"
          size="large"
          style={{
            backgroundColor: severity.color,
            color: theme.palette.common.white
          }}
          sx={{
            '&:hover': {
              backgroundColor: prediction === 1
                ? theme.palette.error.dark
                : theme.palette.primary.dark
            }
          }}
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ResultsDialog;