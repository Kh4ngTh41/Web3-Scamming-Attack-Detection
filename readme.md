# 🚀 End-to-End Web3 Scam Detection System

> 🚧 **Project Status: In-Progress** 🚧
>
> *Currently in the deployment phase of the model via FastAPI.*

This project is an end-to-end system designed to detect and flag fraudulent activities in the Web3 space. It leverages a **Multi-Task Learning (MTL)** model trained on raw on-chain data to provide real-time risk analysis, which is then served via a **FastAPI** backend.

---

## 🎯 Key Features

* **End-to-End Pipeline:** Covers the entire workflow from raw data collection and processing to model deployment.
* **Advanced AI Modeling:** Utilizes a **Multi-Task Learning (MTL)** model (built in **PyTorch**) to simultaneously predict multiple risk factors (e.g., scam classification, wallet address risk).
* **Real-time Inference:** Designed to be deployed as a high-performance REST API using **FastAPI**.
* **Scalable & Actionable:** The API is intended to be the backbone for user-facing tools, such as a browser extension, to warn users in real-time.

---

## 🛠️ Technology Stack

* **Programming Language:** Python
* **AI & ML:** PyTorch, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **API Deployment:** FastAPI, Uvicorn
* **Development:** Jupyter Notebooks, Git

---

## 📈 Methodology

The project follows a structured approach:

1.  **Data Collection:** Gathering raw data from OpenSea.
2.  **Feature Engineering:** Processing and transforming raw data into meaningful features suitable for an ML model.
3.  **Multi-Task Modeling (MTL):** Designing and training a single PyTorch model to solve multiple, related tasks simultaneously, improving generalization and efficiency.
    * **Task 1:** `[Account-level classify]`
    * **Task 2:** `[Transaction-level classify]`

---

## 🚀 Roadmap & Future Development

This project is under active development. The current roadmap is as follows:

* [x] **1. Data Collection & Feature Engineering**
* [x] **2. Multi-Task Model Training & Validation**
* [ ] **3. API Deployment (Current Step):**
    * Finalize and containerize (Docker) the **FastAPI** application to serve the trained model.
* [ ] **4. Extension Integration:**
    * Develop a browser extension (e.g., Chrome) that consumes the API.
    * The extension will analyze the user's current page or transaction request and provide real-time warnings.
* [ ] **5. CI/CD & Monitoring:**
    * Implement a CI/CD pipeline for automated testing and deployment.
    * Set up model monitoring to detect data/model drift.
