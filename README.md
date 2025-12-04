<h1 align="center"> Anomaly Detection Pipeline - Mental Wellness </h1>
<p align="center">A Comprehensive MLOps Framework for Proactive Health Monitoring and Insight Generation.</p>

<p align="center">
  <img alt="Build" src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge">
  <img alt="Contributions" src="https://img.shields.io/badge/Contributions-Welcome-orange?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

---

## 🧭 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack & Architecture](#-tech-stack--architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🌟 Overview

The **Anomaly Detection Pipeline** project delivers a robust, end-to-end Machine Learning Operations (MLOps) framework designed to identify unusual patterns or significant deviations in mental-wellness–related data. Rather than labeling individuals, the system focuses on detecting signals that may suggest elevated stress, emerging mental-health concerns (e.g., early signs of depression or anxiety), or sudden shifts in well-being indicators. This enables proactive monitoring and timely support interventions while maintaining sensitivity and respect for user privacy.


### The Problem

> Deploying and maintaining reliable machine learning models in production, especially for critical tasks like mental wellness, presents significant challenges. Data drift, model decay, and lack of reproducibility are common pain points. Furthermore, building a scalable, high-performance API that integrates complex data preparation, feature engineering (scaling, encoding), and real-time inference often requires disparate tools and extensive manual configuration. Organizations struggle to move beyond experimental models to create production-ready, traceable, and secure prediction services.

### The Solution

This project addresses these challenges by offering a structured MLOps pipeline leveraging industry-standard tools for maximum efficiency and traceability. It separates the ML lifecycle into clear, modular stages—data ingestion, preprocessing, training, evaluation, and serving—ensuring high reproducibility and simplifying maintenance. The core of the solution is a high-performance REST API built with `FastAPI`, capable of delivering real-time anomaly predictions based on input features, making it immediately deployable and scalable.

### Architecture Overview

The pipeline follows a modern, component-based architecture focused on data science best practices. Data handling and model training are managed via dedicated Python modules (`src/data_ingestion.py`, `src/data_preprocessing.py`, `src/model_train.py`). Configuration is externalized (`config/`), and the entire environment is designed for containerization using `Docker`, facilitating seamless deployment. The prediction service operates as a high-performance REST API, utilizing the speed of `FastAPI` for its endpoints, enabling fast and reliable inference at scale.

---

## ✨ Key Features

This system transforms complex data science procedures into reliable, operational services, focusing on core benefits for deployment and monitoring.

### 🚀 High-Performance Real-Time Prediction
Leveraging the power of **FastAPI** and asynchronous programming, the system offers incredibly fast inference.

*   **Fast & Reliable API:** Provides a dedicated, high-throughput `POST /predict` endpoint that processes input features and returns anomaly predictions instantaneously.
*   **Health Monitoring:** Includes a dedicated `GET /health` endpoint for quick status checks, crucial for orchestrators like Kubernetes or Docker Swarm, ensuring zero-downtime deployment.

### 🔄 Fully Modular ML Pipeline
The entire machine learning process is broken down into discrete, testable stages, ensuring high quality and flexibility.

*   **Data Ingestion & Splitting:** Dedicated scripts (`data_ingestion.py`) handle the secure loading, validation, and splitting of raw data into training and testing sets, ensuring data integrity from the start.
*   **Advanced Data Preprocessing:** The pipeline supports multiple transformation techniques, including:
    *   Metadata feature dropping.
    *   Scaling of numeric features (`scale_numeric_features`).
    *   One-Hot Encoding of categorical features (`onehot_encode_features`).
    *   Ordinal and Binary Encoding for specific categorical types (`encode_ordinal_features`, `encoder_binary_features`).
*   **Robust Model Training:** The `model_train.py` script encapsulates the full training lifecycle, from loading preprocessed data to fitting the anomaly detection model.

### 🔬 Reproducibility and Experiment Tracking
The framework is built to support MLOps best practices, ensuring that models and data versions can be tracked and recreated precisely.

*   **Data and Model Versioning (DVC):** Integration with Data Version Control (`dvc.yaml`, `dvc.lock`) ensures that the exact dataset and processing steps used to train any model can be retrieved, guaranteeing reproducibility.
*   **Automated Artifact Logging (MLflow integration):** The model training process automatically logs parameters, metrics, and the final trained model artifact to a centralized tracking server (MLflow, implicitly used due to `mlflow` dependency and `log_artifacts` function in `model_train.py`), allowing for easy comparison and auditing of experiments.
*   **Configuration Management (YAML):** Uses external YAML configuration files (`config/paths.yaml`, `config/columns.yaml`), ensuring all hyperparameters, file paths, and column definitions are managed centrally and dynamically loaded (`load_config` functions).

### 🐳 Containerized Deployment
The project is instantly deployable across any modern infrastructure using Docker.

*   **Dockerfile Included:** A fully configured `Dockerfile` streamlines environment setup, packaging the application and all its dependencies (`requirements.txt`) into a single, portable image, minimizing "works on my machine" issues.

---

## 🛠️ Tech Stack & Architecture

The project leverages a modern Python-centric backend ecosystem focused on high performance, reliable deployment, and data-science integrity.

| Category | Technology | Purpose | Why it was Chosen |
| :--- | :--- | :--- | :--- |
| **Backend API Framework** | `FastAPI` | Powers the high-performance, real-time anomaly prediction API. | Extremely fast, async-driven, strongly typed via Pydantic, and auto-generates interactive API docs (Swagger & ReDoc). |
| **Containerization** | `Docker` | Encapsulates the application, dependencies, and environment. | Ensures consistent behavior across dev, staging, and production while simplifying CI/CD workflows. |
| **Pipeline Versioning** | `dvc` | Manages datasets, model versions, and intermediate pipeline steps. | Guarantees reproducibility, auditability, and reliable MLOps orchestration. |
| **ML / Data Science** | `scikit-learn`, `pandas`, `numpy`<br>Algorithms: **LOF**, **One-Class SVM**, **Isolation Forest** | Provides the core anomaly detection models and data manipulation capabilities. | LOF detects local density deviations, One-Class SVM learns boundary-based anomaly regions, and Isolation Forest offers scalable, tree-based outlier scoring—well-suited for mental-wellness anomaly detection. |
| **Experiment Tracking** | `mlflow` | Tracks experiments, model metrics, parameters, and versions. | Enables systematic comparison of model iterations and maintains a traceable training history. |
| **Artifact Store** | **AWS S3 Bucket** | Stores trained model artifacts, logs, and DVC/MLflow outputs. | Highly durable, scalable, and cloud-native—perfect for MLOps pipelines and distributed workflows. |


---

## 📁 Project Structure

The repository is organized following best practices for MLOps projects, separating source code, configurations, data, and continuous integration assets.

```
austinthapa-anomaly-detection-system-0346066/
├── 📄 .gitignore                 # Specifies files and directories to be ignored by Git
├── 📄 app.py                     # Main FastAPI application entry point, including prediction logic
├── 📄 dvc.lock                   # DVC lock file, tracks exact versions of data/models used
├── 📄 dvc.yaml                   # DVC pipeline definition file (steps for data/model lifecycle)
├── 📄 Dockerfile                 # Configuration file for building the application container
├── 📄 LICENSE                    # Project license (MIT)
├── 📄 pytest.ini                 # Configuration for the Pytest testing framework
├── 📄 README.md                  # Project documentation (this file)
├── 📄 requirements.txt           # Python dependency list (used for environment setup)
├── 📂 .github/                   
│   └── 📂 workflows/             # Definitions for GitHub Actions CI/CD
│       └── 📄 ci-cd.yaml         # Continuous Integration and Continuous Deployment pipeline
├── 📂 config/                    # Centralized configuration files
│   ├── 📄 columns.yaml           # Defines column names (e.g., numeric, categorical, metadata)
│   └── 📄 paths.yaml             # Defines file paths for data, models, and outputs
├── 📂 data/                      
│   └── 📄 .gitignore             # Ignores large data files from Git, deferring to DVC
├── 📂 src/                       
│   ├── 📄 __init__.py            # Marks src as a Python package
│   ├── 📄 data_ingestion.py      # Script for loading, validating, and splitting raw data
│   ├── 📄 data_preprocessing.py  # Script for feature engineering (scaling, encoding, dropping)
│   └── 📄 model_train.py         # Script for model training, evaluation, and artifact logging
└── 📂 tests/                     
    ├── 📄 test_data_ingestion.py      # Tests for data loading and splitting logic
    ├── 📄 test_data_preprocessing.py  # Tests for feature scaling and encoding
    └── 📄 test_model_train.py         # Tests for model configuration and training process
```

---

## 🚀 Getting Started

To run the Anomaly Detection Pipeline locally or deploy it via Docker, follow these steps.

### Prerequisites

You will need the following software installed on your system:

*   **Python:** Version 3.8+ (required for dependencies in `requirements.txt`).
*   **pip:** The Python package installer.
*   **Docker:** (Recommended) For building and running the containerized application.

### Installation and Setup

#### Option 1: Using Docker (Recommended for Production)

The fastest and most reliable way to run the application is using the provided `Dockerfile`.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/austinthapa/anomaly-detection-system.git
    cd anomaly-detection-system
    ```

2.  **Build the Docker Image:**
    This command reads the `Dockerfile` and creates a container image named `anomaly-detector-api`.
    ```bash
    docker build -t anomaly-detector-api .
    ```

3.  **Run the Container:**
    Run the image, mapping the internal port (e.g., 8000, default for FastAPI) to an external port on your host machine (e.g., 8080).
    ```bash
    docker run -d -p 8080:8000 --name anomaly_service anomaly-detector-api
    ```
    The service will now be running at `http://localhost:8080`.

#### Option 2: Local Python Environment

1.  **Clone the Repository (if not already done):**
    ```bash
    git clone https://github.com/austinthapa/anomaly-detection-system.git
    cd anomaly-detection-system
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv anomaly-venv
    source anomaly-venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required packages listed in `requirements.txt`. Note that this includes complex data science tools like `dvc`, `mlflow`, `scikit-learn`, and the API server (`fastapi`, `uvicorn`).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Prediction API:**
    The entry point is `app.py`, which is a FastAPI application that handles prediction serving.
    ```bash
    # Assuming the server runs on the standard FastAPI/Uvicorn port
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    The API should now be available at `http://127.0.0.1:8000`.

### Running the ML Pipeline (Data and Model Training)

Before serving predictions, the anomaly detection model must be trained. This process is managed by the pipeline scripts in the `src/` directory.

1.  **Configure Paths and Columns:**
    Ensure `config/paths.yaml` and `config/columns.yaml` are correctly set up to point to your data files and define the features used for training.

2.  **Execute the Pipeline Stages:**
    The scripts can be run sequentially to simulate the full MLOps workflow:

    *   **Data Ingestion and Splitting:**
        ```bash
        python src/data_ingestion.py
        ```
    *   **Data Preprocessing (Scaling and Encoding):**
        ```bash
        python src/data_preprocessing.py
        ```
    *   **Model Training and Evaluation:**
        ```bash
        python src/model_train.py
        ```
    *   *Note: These steps will automatically log parameters and artifacts to MLflow if configured.*

### Running Tests

Ensure all components are functioning correctly using the dedicated unit tests.

```bash
pytest tests/
```
The testing suite covers ingestion, preprocessing logic (scaling, encoding), data saving, configuration loading, and model training processes, guaranteeing component stability before deployment.

---

## 🔧 Usage

This project provides a fully functional RESTful API designed for integrating anomaly detection capabilities into external applications or dashboards.

### API Endpoints

The service exposes three primary endpoints, all powered by `FastAPI`. Assuming the service is running locally on port 8000.

| Endpoint | Method | Description | Functionality |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Root endpoint. | Returns a simple confirmation message or service information. |
| `/health` | `GET` | Health Check. | Used by monitoring tools to determine if the service is operational and responsive. |
| `/predict` | `POST` | Anomaly Prediction. | Core functionality: accepts structured input data and returns an anomaly score or classification. |

### 1. Health Check (`GET /health`)

Use this endpoint to confirm that the API service is running correctly.

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response (Example):**
```json
{
  "status": "ok"
}
```

### 2. Making a Prediction (`POST /predict`)

This is the main prediction interface. It requires structured input data conforming to the `PredictionInput` schema defined in `app.py`. The input must contain the specific features required by the trained anomaly detection model.

**Input Schema (`PredictionInput`):**
The payload must be a JSON object where the keys correspond to the model's expected feature names (e.g., numeric, one-hot, ordinal, and binary features).

**Request:**
Send a JSON payload to the `/predict` endpoint.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "number_of_children": 2,
    "income": 50000,
    "marital_status": "Married",
    "smoking_status": "Non-smoker",
    "physical_activity_level": "Moderate",
    "education_level": "Bachelor'\''s Degree",
    "alcohol_consumption": "Moderate",
    "dietary_habits": "Healthy",
    "sleep_patterns": "Good",
    "employment_status": "Employed",
    "history_of_mental_illness": "No",
    "history_of_substance_abuse": "No",
    "family_history_of_depression": "No",
    "chronic_medical_conditions": "No"
  }'
```

**Response (Example):**
The response will indicate the outcome of the anomaly detection for the provided input. This typically involves an anomaly score or a binary classification (anomaly/normal).

```json
{
  "prediction": 1,
  "confidence_score": 0.985,
  "message": "Anomaly Detected"
}
```

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.
