import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box, Container, Stack } from '@mui/material';
import MLWorkflow from './MLWorkflow';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function LandingPage() {
  const navigate = useNavigate();
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Container maxWidth="md" sx={{ py: 8 }}>
        <Stack spacing={6} alignItems="center">
          <Typography variant="h2" fontWeight={800} color="primary.main" align="center">
            ML Model as a Service
          </Typography>
          <Typography variant="h5" color="text.secondary" align="center">
            Upload or link your dataset, train models, auto-version, deploy, and manage everything from a beautiful dashboard.<br />
            <b>Production-grade MLOps, simplified.</b>
          </Typography>
          <Button
            size="large"
            variant="contained"
            color="primary"
            endIcon={<RocketLaunchIcon />}
            sx={{ fontWeight: 700, px: 6, py: 2, fontSize: 22 }}
            onClick={() => navigate('/app')}
          >
            Get Started
          </Button>
          <Stack spacing={2} alignItems="center" sx={{ width: '100%' }}>
            <Typography variant="h6" color="text.secondary">Try with a sample dataset:</Typography>
            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                color="secondary"
                href="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
                target="_blank"
                startIcon={<CloudUploadIcon />}
              >
                Boston Housing (Regression)
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                href="https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
                target="_blank"
                startIcon={<CloudUploadIcon />}
              >
                Iris (Classification)
              </Button>
            </Stack>
          </Stack>
          <Box sx={{ mt: 8 }}>
            <Typography variant="h4" fontWeight={700} color="primary.main" align="center" gutterBottom>
              Features
            </Typography>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={4} justifyContent="center" alignItems="center">
              <FeatureCard title="AutoML" desc="Automatic model selection, training, and evaluation." />
              <FeatureCard title="Versioning" desc="Every model version is tracked and managed." />
              <FeatureCard title="Deployment" desc="Deploy any model version to a live API endpoint." />
              <FeatureCard title="Dashboard" desc="Monitor, rollback, and manage all your models." />
            </Stack>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}

function FeatureCard({ title, desc }: { title: string; desc: string }) {
  return (
    <Box sx={{ p: 3, bgcolor: 'white', borderRadius: 3, boxShadow: 2, minWidth: 220, textAlign: 'center' }}>
      <Typography variant="h6" fontWeight={700} color="primary.main" gutterBottom>{title}</Typography>
      <Typography variant="body1" color="text.secondary">{desc}</Typography>
    </Box>
  );
}

function App() {
  return (
    <Router>
      <AppBar position="static" color="primary" elevation={2}>
        <Toolbar>
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: 1 }}>
            <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
              ML Model as a Service
            </Link>
          </Typography>
          <Button color="inherit" component={Link} to="/app" sx={{ fontWeight: 600 }}>
            App
          </Button>
        </Toolbar>
      </AppBar>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<MLWorkflow />} />
      </Routes>
    </Router>
  );
}

export default App;
