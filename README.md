# MLOps-Sentinel: Predictive Service & Monitoring Pipeline

**Author: Adam Silva** **GitHub:** [https://github.com/adaumsilva/](https://github.com/adaumsilva/)  
**Specialization:** AI Engineering, RAG Architectures, and MLOps.

---

MLOps-Sentinel is a production-grade machine learning repository designed to demonstrate the complete lifecycle of a predictive model. Moving beyond experimental notebooks, this project implements a robust engineering framework for training, deploying, and monitoring high-availability models in a containerized environment.

---

## Key Features

* **Automated Training Pipeline:** A modularized Python engine for data preprocessing, feature engineering, and Scikit-learn/XGBoost model training.
* **Production API Layer:** An asynchronous FastAPI implementation providing a /predict endpoint with strict Pydantic data validation.
* **Containerization:** Optimized multi-stage Docker builds to ensure consistent environments and small image footprints.
* **CI/CD Integration:** Automated GitHub Actions workflows for code linting (Black/Flake8) and unit testing on every push.
* **Observability:** Integration with Prometheus metrics to track inference latency and request volume in real-time.

---

## System Architecture

The project follows a modular MLOps structure:

1.  **Data Ingestion:** Raw data is processed through versioned transformation scripts.
2.  **Model Training:** Models are trained and serialized using standard persistence formats.
3.  **Deployment:** The API serves the model via a high-performance Uvicorn server.
4.  **Monitoring:** Metrics are exposed via a /metrics endpoint for Prometheus scraping.

---

## Tech Stack

* **Model Framework:** Scikit-learn / XGBoost
* **API Framework:** FastAPI / Uvicorn
* **Validation:** Pydantic
* **Infrastructure:** Docker / Docker-compose
* **CI/CD:** GitHub Actions
* **Monitoring:** Prometheus / Client_python
* **Quality Assurance:** Pytest

---

## Getting Started

### Prerequisites
* Python 3.10 or higher
* Docker and Docker Compose

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/adaumsilva/MLOps-Sentinel.git](https://github.com/adaumsilva/MLOps-Sentinel.git)
    cd MLOps-Sentinel
    ```

2.  Set up the environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  Train the initial model:
    ```bash
    python src/training/train.py
    ```

### Running with Docker

To deploy the API and Prometheus monitoring stack simultaneously:
```bash
docker-compose up --build