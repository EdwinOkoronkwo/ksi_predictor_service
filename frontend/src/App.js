import React from 'react';
import { CssBaseline, Container } from '@mui/material';
import FeatureInputForm from "./components/FeatureInputForm/FeatureInputForm";
import theme from "./styles/theme";
import {ThemeProvider} from "@mui/material/styles";

function App() {
  return (
     <ThemeProvider theme={theme}>
          <CssBaseline />
          <Container maxWidth={false} disableGutters>
              <FeatureInputForm />
          </Container>
     </ThemeProvider>
  );
}

export default App;