# Backend - ML Model as a Service

## Setup

1. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the API

```sh
uvicorn main:app --reload
```

## MLflow UI

To launch the MLflow tracking UI (optional):
```sh
mlflow ui
```

The API will be available at http://localhost:8000
The MLflow UI will be available at http://localhost:5000 