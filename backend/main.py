from fastapi import FastAPI, File, UploadFile, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
import uuid
import pickle
import json
import shutil
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_csvs"
STATIC_DIR = "static"
EDA_DIR = os.path.join(STATIC_DIR, "eda")
MODEL_DIR = "models"
META_FILE = "models/meta.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
if not os.path.exists(META_FILE):
    with open(META_FILE, "w") as f:
        json.dump([], f)

def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def load_meta():
    with open(META_FILE, "r") as f:
        return json.load(f)

def process_data(filepath):
    data = pd.read_csv(filepath)
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna("Unknown", inplace=True)
    target_col = data.columns[-1]
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    target_type = "classification" if data[target_col].nunique() < 15 else "regression"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    data_summary = data.describe(include='all').transpose().to_dict()
    heatmap_path = os.path.join(STATIC_DIR, 'heatmap.png')
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    return data_summary, heatmap_path, X_train, X_test, y_train, y_test, target_type

def perform_eda(filepath):
    data = pd.read_csv(filepath)
    plots = []
    save_dir = EDA_DIR
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        plot_path = os.path.join(save_dir, f"{column}_hist.png")
        sns.histplot(data[column], kde=True, color='blue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots.append(plot_path)
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        plot_path = os.path.join(save_dir, f"{column}_boxplot.png")
        sns.boxplot(x=data[column], color='orange')
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots.append(plot_path)
    if len(data.columns) <= 5:
        pair_plot_path = os.path.join(save_dir, "pair_plot.png")
        sns.pairplot(data)
        plt.tight_layout()
        plt.savefig(pair_plot_path)
        plt.close()
        plots.append(pair_plot_path)
    if data.select_dtypes(include='object').shape[1] > 0:
        for col in data.select_dtypes(include='object').columns:
            count_plot_path = os.path.join(save_dir, f"{col}_count.png")
            sns.countplot(y=data[col], palette="viridis")
            plt.title(f"Count Plot of {col}")
            plt.tight_layout()
            plt.savefig(count_plot_path)
            plt.close()
            plots.append(count_plot_path)
    return plots

def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results, target_type):
    metric = "accuracy" if target_type == "classification" else "r2_score"
    model_names = list(results.keys())
    scores = [results[model][metric] for model in model_names]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=scores, palette='viridis')
    plt.title(f'Model Comparison ({metric.capitalize()})')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    chart_path = os.path.join(STATIC_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_report(data_summary, eda_plots, model_results, confusion_matrices, model_comparison_chart, version):
    report_path = os.path.join(STATIC_DIR, f'{version}_report.pdf')
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.set_font('Arial', 'B', size=16)
    pdf.cell(200, 10, f'Data Analysis Report - {version}', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', 'B', size=14)
    pdf.cell(200, 10, 'Data Summary', ln=True, align='L')
    pdf.set_font('Arial', size=12)
    if data_summary:
        for key, value in data_summary.items():
            pdf.cell(200, 10, f"{key}:", ln=True)
            pdf.multi_cell(0, 10, f"  {value}", align='L')
            pdf.ln(5)
    else:
        pdf.cell(200, 10, "No data summary available.", ln=True)
    if eda_plots:
        for plot in eda_plots:
            pdf.add_page()
            pdf.image(plot, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No EDA plots available.", ln=True)
    pdf.add_page()
    pdf.set_font('Arial', 'B', size=14)
    pdf.cell(200, 10, 'Model Results', ln=True, align='L')
    pdf.set_font('Arial', size=12)
    if model_results:
        for model, report in model_results.items():
            pdf.cell(200, 10, f"Model: {model}", ln=True)
            for metric, score in report.items():
                pdf.cell(200, 10, f"  {metric}: {score}", ln=True)
            pdf.ln(5)
    else:
        pdf.cell(200, 10, "No model results available.", ln=True)
    if confusion_matrices:
        for model, cm_path in confusion_matrices.items():
            pdf.add_page()
            pdf.cell(200, 10, f"Confusion Matrix for {model}", ln=True, align='C')
            pdf.image(cm_path, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No confusion matrices available.", ln=True)
    if model_comparison_chart:
        pdf.add_page()
        pdf.cell(200, 10, 'Model Comparison', ln=True, align='C')
        pdf.image(model_comparison_chart, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No model comparison chart available.", ln=True)
    pdf.output(report_path)
    return report_path

@app.post("/upload_csv")
async def upload_csv(request: Request, file: UploadFile = File(None)):
    if file is not None:
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        columns = list(df.columns)
        sample_rows = df.head(10).to_dict(orient="records")
        missing = df.isnull().sum().to_dict()
        return {
            "columns": columns,
            "sample_rows": sample_rows,
            "missing": missing,
            "filename": filename
        }
    try:
        data = await request.json()
        url = data.get('url')
        if url:
            file_id = str(uuid.uuid4())
            filename = f"{file_id}_from_url.csv"
            file_path = os.path.join(UPLOAD_DIR, filename)
            import requests as pyrequests
            r = pyrequests.get(url)
            with open(file_path, "wb") as f:
                f.write(r.content)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
            columns = list(df.columns)
            sample_rows = df.head(10).to_dict(orient="records")
            missing = df.isnull().sum().to_dict()
            return {
                "columns": columns,
                "sample_rows": sample_rows,
                "missing": missing,
                "filename": filename
            }
    except Exception:
        pass
    return JSONResponse(status_code=400, content={"error": "No file or valid URL provided."})

@app.post("/train")
def train_model(filename: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    data_summary, heatmap_path, X_train, X_test, y_train, y_test, target_type = process_data(file_path)
    eda_plots = perform_eda(file_path)
    results = {}
    confusion_matrices = {}
    models = []
    if target_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=500)),
            ("Random Forest", RandomForestClassifier(n_estimators=100)),
            ("Support Vector Machine", SVC()),
        ]
        metric = "accuracy"
    else:
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
            ("Support Vector Regressor", SVR()),
        ]
        metric = "r2_score"
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if target_type == "classification":
            score = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            cm_path = os.path.join(STATIC_DIR, f"{name.replace(' ', '_')}_confusion_matrix.png")
            plot_confusion_matrix(cm, name, cm_path)
            confusion_matrices[name] = cm_path
        else:
            score = r2_score(y_test, predictions)
        results[name] = {metric: round(score, 4)}
    comparison_chart = plot_model_comparison(results, target_type)
    best_model_name = max(results, key=lambda k: results[k][metric])
    best_model_idx = [m[0] for m in models].index(best_model_name)
    best_model = models[best_model_idx][1]
    version = f"v{len(load_meta())+1}"
    model_file = os.path.join(MODEL_DIR, f"{version}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    meta = load_meta()
    meta.append({
        "version": version,
        "metrics": results[best_model_name],
        "target_type": target_type,
        "filename": filename,
        "deployed": False,
        "best_model": best_model_name
    })
    save_meta(meta)
    report_path = generate_report(data_summary, eda_plots, results, confusion_matrices, comparison_chart, version)
    return {
        "status": "success",
        "metrics": results[best_model_name],
        "model_version": version,
        "training_time": "few seconds",
        "eda_plots": eda_plots,
        "heatmap": heatmap_path,
        "confusion_matrices": confusion_matrices,
        "comparison_chart": comparison_chart,
        "report_path": report_path,
        "best_model": best_model_name
    }

@app.get("/models")
def list_models():
    return load_meta()

@app.get("/download_model")
def download_model(version: str = Query(...)):
    model_file = os.path.join(MODEL_DIR, f"{version}.pkl")
    if not os.path.exists(model_file):
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    return FileResponse(model_file, filename=f"{version}.pkl")

@app.get("/download_report")
def download_report(version: str = Query(...)):
    report_file = os.path.join(STATIC_DIR, f"{version}_report.pdf")
    if not os.path.exists(report_file):
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return FileResponse(report_file, filename=f"{version}_report.pdf")

@app.post("/deploy")
def deploy_model(version: str = Form(...)):
    meta = load_meta()
    for m in meta:
        m["deployed"] = (m["version"] == version)
    save_meta(meta)
    return {"status": "deployed", "version": version}

@app.post("/rollback")
def rollback_model(version: str = Form(...)):
    meta = load_meta()
    for m in meta:
        m["deployed"] = (m["version"] == version)
    save_meta(meta)
    return {"status": "rolled back", "version": version}