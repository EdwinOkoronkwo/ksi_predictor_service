import React from 'react';
import { CssBaseline, Container } from '@mui/material';
import aiImage from "../src/assets/ai-image.jpg";
import FeatureInputForm from "./components/FeatureInputForm";

function App() {
  return (
    <>
      <CssBaseline />
      <Container
        maxWidth={false}
        disableGutters
        sx={{
          backgroundImage: `url(${aiImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          backgroundAttachment: 'fixed',
            height: '100vh'
        }}
      >

          <FeatureInputForm />
      </Container>
    </>
  );
}

export default App;