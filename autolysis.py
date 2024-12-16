# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "chardet",
#   "scipy",
#   "scikit-learn",
#   "python-dotenv"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

if not find_dotenv():
    print("Warning: .env file not found. Ensure AIPROXY_TOKEN is set.")

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1"


def create_or_use_directory(dir_name):
    """Create a directory if it doesn't exist, or use the existing one."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created.")
    else:
        print(f"Directory '{dir_name}' already exists.")
    return dir_name


def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)


def analyze_data(df):
    """Perform generic data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)

    # Summary statistics for all columns
    summary = df.describe(include='all').to_dict()

    # Counting missing values
    missing_values = df.isnull().sum().to_dict()

    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr().to_dict()

    # Outlier detection using Z-score for numeric columns
    z_scores = numeric_df.apply(stats.zscore, nan_policy='omit')
    outliers = {col: list(numeric_df.index[abs(z_scores[col]) > 3]) for col in numeric_df.columns}

    # Clustering (e.g., KMeans) for numeric columns
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df.fillna(0))  # Replace NaNs with 0 or another strategy
    kmeans = KMeans(n_clusters=3, random_state=42).fit(numeric_scaled)
    
    # Assign clusters to the original DataFrame
    df['cluster'] = pd.Series(kmeans.labels_, index=numeric_df.index)

    # Consolidating analysis results
    analysis = {
        'summary': summary,
        'missing_values': missing_values,
        'correlation': correlation,
        'outliers': outliers,
        'clustering': df[['cluster']].head().to_dict()
    }
    
    print("Data analysis complete.")
    return analysis


def visualize_data(df, output_dir):
    """Generate and save visualizations for all numeric features."""
    sns.set(style="whitegrid")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Check if there are numeric features to visualize
    if numeric_df.empty:
        print("No numeric features found. Skipping visualizations.")
        return  # Skip visualizations if no numeric features

    # Correlation heatmap for all numeric features
    plt.figure(figsize=(14, 10))
    correlation_matrix = numeric_df.corr()
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap for Numeric Features')
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        print(f"Saved correlation heatmap: {heatmap_path}")
        plt.close()

    # Pairplot for all numeric features with clustering (if applicable)
    if 'cluster' in df.columns:
        num_clusters = df['cluster'].nunique()
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'h', '*'][:num_clusters]
        
        sampled_df = df.sample(1000, random_state=42) if len(df) > 1000 else df
        pairplot = sns.pairplot(sampled_df, vars=numeric_df.columns, hue='cluster', palette='viridis', markers=markers)
        pairplot_path = os.path.join(output_dir, 'pairplot_numeric_features.png')
        pairplot.savefig(pairplot_path)
        print(f"Saved pairplot: {pairplot_path}")
        plt.close()

        # Scatter plot for clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[numeric_df.columns[0]], y=df[numeric_df.columns[1]], hue=df['cluster'], palette='viridis')
        plt.title('Clustered Data Points')
        scatter_plot_path = os.path.join(output_dir, 'cluster_scatter_plot.png')
        plt.savefig(scatter_plot_path)
        print(f"Saved cluster scatter plot: {scatter_plot_path}")
        plt.close()


def generate_narrative(analysis, file_path, df, output_dir):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }

    # Prepare data for LLM
    column_info = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
    context = {
        'file_name': file_path,
        'columns': column_info,
        'summary_statistics': analysis['summary'],
        'missing_values': analysis['missing_values'],
        'correlation_matrix': analysis['correlation'],
        'outliers': analysis['outliers'],
        'clustering_preview': analysis['clustering']
    }

    prompt = f"Analyze the following data and provide insights and suggestions for further analysis: {context}"

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in narrative generation: {e}")
        return "Narrative generation failed due to an error."


def main(file_path):
    print("Starting autolysis process...")
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = create_or_use_directory(base_name)
    
    print("Loading dataset...")
    df = load_data(file_path)
    print("Dataset loaded successfully.")
    
    print("Analyzing data...")
    analysis = analyze_data(df)
    
    print("Generating visualizations...")
    visualize_data(df, output_dir)  # Now generates visualizations for all features
    
    print("Generating narrative...")
    narrative = generate_narrative(analysis, file_path, df, output_dir)
    
    if narrative != "Narrative generation failed due to an error.":
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write("### Overview\n")
            f.write(narrative)
            f.write("\n\n### Data Visualizations\n")
            f.write("#### Correlation Heatmap\n")
            f.write(f"![Correlation Heatmap](correlation_heatmap.png)\n")
            f.write("#### Pairplot\n")
            f.write(f"![Pairplot](pairplot_numeric_features.png)\n")
            f.write("#### Cluster Scatter Plot\n")
            f.write(f"![Cluster Scatter Plot](cluster_scatter_plot.png)\n")
        print(f"Narrative successfully written to {readme_path}.")
    else:
        print("Narrative generation failed. Skipping README creation.")
    
    print("Autolysis process completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
