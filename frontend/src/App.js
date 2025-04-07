import React from 'react';
import { CssBaseline, Container } from '@mui/material';
import aiImage from "../src/assets/ai-image.jpg";
import FeatureInputForm from "./components/FeatureInputForm";

function App() {
  return (
    <>
      <CssBaseline />
      <Container maxWidth={false} disableGutters>
          <FeatureInputForm />
      </Container>
    </>
  );
}

export default App;