import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Environment variable AIPROXY_TOKEN is not set. Please check your .env file.")

API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"

# Set up logging
logging.basicConfig(
    filename="autolysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Create output directory
def create_output_directory(dataset_name):
    directory = dataset_name.split('.')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Load dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, encoding="latin1")
        logging.info("Dataset loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

# Preprocess data
def preprocess_data(data):
    try:
        data = data.apply(pd.to_numeric, errors="coerce")  # Convert all applicable columns to numeric
        data.fillna(data.mean(numeric_only=True), inplace=True)  # Fill numeric NaNs with mean
        data.fillna("Unknown", inplace=True)  # Fill non-numeric NaNs with "Unknown"
        logging.info("Data preprocessing completed.")
        return data
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {e}")

# Analyze data
def analyze_data(data):
    try:
        numeric_cols = data.select_dtypes(include=["number"])
        categorical_cols = data.select_dtypes(exclude=["number"])

        summary = numeric_cols.describe().to_string()
        missing = data.isnull().sum()
        correlation = numeric_cols.corr()

        categorical_summary = categorical_cols.describe(include="all").to_string() if not categorical_cols.empty else "No categorical data."

        logging.info("Data analysis completed.")
        return {
            "summary": summary,
            "missing": missing,
            "correlation": correlation,
            "categorical_summary": categorical_summary,
        }
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {e}")

# Visualize data
def visualize_data(data, correlation, output_dir):
    try:
        if not correlation.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.savefig(os.path.join(output_dir, f"correlation_heatmap.png"))
            plt.close()

        numeric_cols = data.select_dtypes(include=["number"])
        for col in numeric_cols.columns:
            plt.figure()
            sns.histplot(numeric_cols[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.savefig(os.path.join(output_dir, f"{col}_histogram.png"))
            plt.close()

        logging.info("Visualizations saved as PNG.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")

# Generate narrative using AI Proxy
def generate_narrative(data_info):
    try:
        prompt = f"""
        You are a data analyst. Here is the dataset summary:
        {data_info['summary']}
        Missing values:
        {data_info['missing']}
        Categorical data summary:
        {data_info['categorical_summary']}
        Correlation analysis (if applicable):
        {data_info['correlation']}

        Write a detailed analysis story with insights, implications, and recommendations.
        """
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        narrative = response.json()["choices"][0]["message"]["content"]
        logging.info("Narrative generated successfully.")
        return narrative
    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate narrative: {e}")

# Create README.md
def create_readme(narrative, output_dir):
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as file:
            file.write("# Analysis Report\n\n")
            file.write(narrative)
            file.write("\n\n![Correlation Heatmap](correlation_heatmap.png)\n")
        logging.info("README.md created successfully.")
    except Exception as e:
        logging.error(f"Error creating README.md: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create README.md: {e}")

# API endpoint to trigger analysis (hardcoded dataset path)
@app.post("/analyze")
def analyze():
    try:
        # Hardcoded dataset path
        dataset_path = "E:/tds proj2/goodreads.csv"
        dataset_name = os.path.basename(dataset_path)
        output_dir = create_output_directory(dataset_name)

        data = load_dataset(dataset_path)
        processed_data = preprocess_data(data)
        analysis_results = analyze_data(processed_data)
        visualize_data(processed_data, analysis_results["correlation"], output_dir)
        narrative = generate_narrative(analysis_results)
        create_readme(narrative, output_dir)

        return {"message": f"Analysis completed for {dataset_name}. Output in {output_dir}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Autolysis API!"}
