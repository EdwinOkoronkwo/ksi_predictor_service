import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: { 
      main: '#003366',
      light: '#e6f0fa',
      dark: '#002244',
      contrastText: '#fff'
    },
    secondary: {
      main: '#cc0000',
    },
    error: {
      main: '#d32f2f',
      light: '#fae6e6',
      dark: '#a50a0a'
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff'
    },
  },
  typography: {
    fontFamily: '"TPS Sans", Roboto, "Helvetica Neue", Arial, sans-serif',
    h5: {
      fontWeight: 700,
      fontSize: '1.5rem',
      color: '#003366'
    },
    body2: {
      color: '#555555'
    }
  },
  components: {
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: '16px',
          overflow: 'hidden'
        }
      }
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          borderBottom: '1px solid #e0e0e0',
          paddingBottom: '16px'
        }
      }
    },
    MuiDialogContent: {
      styleOverrides: {
        root: {
          backgroundColor: '#f5f5f5',
          padding: '24px'
        }
      }
    },
    MuiDialogActions: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          borderTop: '1px solid #e0e0e0',
          padding: '16px'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '4px',
          padding: '8px 22px'
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }
      }
    }
  }
});

export default theme;