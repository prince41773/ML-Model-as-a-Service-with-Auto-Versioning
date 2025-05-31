import React, { useState } from 'react';
import { Box, Stepper, Step, StepLabel, Paper, CircularProgress, Stack, Alert, MenuItem, Select, InputLabel, FormControl, Button, TextField, Typography } from '@mui/material';
import { FilePond, registerPlugin } from 'react-filepond';
import 'filepond/dist/filepond.min.css';
import axios from 'axios';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';

registerPlugin();

const steps = ['Upload or Link Dataset', 'Preview & Select Target', 'Train Model'];

function MLWorkflow() {
  const [activeStep, setActiveStep] = useState(0);
  const [file, setFile] = useState<any>(null);
  const [datasetUrl, setDatasetUrl] = useState('');
  const [preview, setPreview] = useState<any>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState('');
  const [problemType, setProblemType] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState<any>(null);
  const [modelVersion, setModelVersion] = useState('');
  const [trainingTime, setTrainingTime] = useState('');

  // FilePond expects a specific server.process signature
  const filePondServer: any = {
    process: (
      fieldName: string,
      file: any, // Accept FilePond's ActualFileObject
      metadata: any,
      load: (p: string) => void,
      error: (e: string) => void,
      progress: (isLengthComputable: boolean, loaded: number, total: number) => void,
      abort: () => void
    ) => {
      setLoading(true);
      setError('');
      const formData = new FormData();
      formData.append('file', file);
      axios.post('http://localhost:8000/upload_csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          progress(e.lengthComputable, e.loaded, e.total ?? 0);
        },
      })
        .then(res => {
          setPreview(res.data);
          setColumns(res.data.columns);
          setActiveStep(1);
          load(res.data.filename);
        })
        .catch(() => {
          setError('Failed to upload or parse CSV.');
          setPreview(null);
          setColumns([]);
          error('Failed to upload or parse CSV.');
        })
        .finally(() => {
          setLoading(false);
        });
      return {
        abort: () => {
          abort();
        },
      };
    },
  };

  // Handle dataset URL submission
  const handleUrlLoad = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await axios.post('http://localhost:8000/upload_csv', { url: datasetUrl });
      setPreview(res.data);
      setColumns(res.data.columns);
      setActiveStep(1);
    } catch (err) {
      setError('Failed to load or parse dataset from URL.');
      setPreview(null);
      setColumns([]);
    } finally {
      setLoading(false);
    }
  };

  // Start training
  const handleTrain = async () => {
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('filename', preview.filename);
      formData.append('target', target);
      formData.append('problem_type', problemType);
      const res = await axios.post('http://localhost:8000/train', formData);
      setMetrics(res.data.metrics);
      setModelVersion(res.data.model_version);
      setTrainingTime(res.data.training_time);
      setActiveStep(2);
    } catch (err: any) {
      setError('Training failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ py: 4 }}>
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      <Paper sx={{ p: 4, minHeight: 400, maxWidth: 600, mx: 'auto' }} elevation={3}>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {activeStep === 0 && (
          <Paper
            elevation={4}
            sx={{
              p: 4,
              mb: 2,
              maxWidth: 500,
              mx: "auto",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              borderRadius: 3,
              bgcolor: "#fafbfc",
            }}
          >
            <Typography variant="h6" sx={{ mb: 2 }}>
              Upload a CSV file
            </Typography>
            <Box sx={{ width: "100%", mb: 2 }}>
              <FilePond
                files={file ? [file] : []}
                onupdatefiles={fileItems => setFile(fileItems[0]?.file || null)}
                allowMultiple={false}
                server={filePondServer}
                name="file"
                labelIdle='<span style="font-size:1.1em;">Drag & Drop your CSV or <span class="filepond--label-action">Browse</span></span>'
                acceptedFileTypes={["text/csv"]}
                maxFiles={1}
                stylePanelLayout="integrated"
                stylePanelAspectRatio="1.5"
              />
            </Box>
            <Typography variant="body1" color="text.secondary" sx={{ my: 2 }}>
              or
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ width: "100%" }}>
              <TextField
                label="Dataset URL"
                variant="outlined"
                value={datasetUrl}
                onChange={e => setDatasetUrl(e.target.value)}
                fullWidth
                size="small"
              />
              <Button
                variant="contained"
                color="secondary"
                endIcon={<CloudDownloadIcon />}
                onClick={handleUrlLoad}
                disabled={!datasetUrl || loading}
                sx={{ minWidth: 120, fontWeight: 600 }}
              >
                Load
              </Button>
            </Stack>
          </Paper>
        )}
        {activeStep === 1 && preview && (
          <Stack spacing={3}>
            <Typography variant="h6">Preview</Typography>
            <Box sx={{ overflowX: 'auto', maxHeight: 300, borderRadius: 2, border: '1px solid #eee', boxShadow: 1 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', background: 'white' }}>
                <thead style={{ background: '#f4f6fa' }}>
                  <tr>
                    {columns.map(col => (
                      <th key={col} style={{ border: '1px solid #ccc', padding: 6, fontWeight: 700 }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.sample_rows.map((row: any, idx: number) => (
                    <tr key={idx}>
                      {columns.map(col => (
                        <td key={col} style={{ border: '1px solid #eee', padding: 6 }}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel id="target-label">Target Column</InputLabel>
              <Select
                labelId="target-label"
                value={target}
                label="Target Column"
                onChange={e => setTarget(e.target.value)}
              >
                {columns.map(col => (
                  <MenuItem key={col} value={col}>{col}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel id="problem-type-label">Problem Type</InputLabel>
              <Select
                labelId="problem-type-label"
                value={problemType}
                label="Problem Type"
                onChange={e => setProblemType(e.target.value)}
              >
                <MenuItem value="classification">Classification</MenuItem>
                <MenuItem value="regression">Regression</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="contained"
              color="primary"
              disabled={!target || !problemType || loading}
              onClick={handleTrain}
              sx={{ mt: 2, fontWeight: 600 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Train Model'}
            </Button>
          </Stack>
        )}
        {activeStep === 2 && (
          <Stack spacing={3} alignItems="center" justifyContent="center" sx={{ minHeight: 300 }}>
            <Typography variant="h4" color="primary">Training Complete!</Typography>
            {metrics && (
              <Box sx={{ width: "100%", maxWidth: 500, bgcolor: "#f4f6fa", p: 3, borderRadius: 2 }}>
                <Typography variant="h6" color="text.secondary">Model Metrics</Typography>
                <ul>
                  {Object.entries(metrics).map(([k, v]) => (
                    <li key={k}>{k}: {v as string}</li>
                  ))}
                </ul>
              </Box>
            )}
            <Stack direction="row" spacing={2}>
              <Button variant="contained" color="primary">Download Model</Button>
              <Button variant="outlined" color="secondary">Download Report</Button>
              <Button variant="contained" color="success">Deploy</Button>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Model Version: {modelVersion} | Training Time: {trainingTime}
            </Typography>
          </Stack>
        )}
      </Paper>
    </Box>
  );
}

export default MLWorkflow;