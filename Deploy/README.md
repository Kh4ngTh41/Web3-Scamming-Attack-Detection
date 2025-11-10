# Web3 Scam Detection API

This API service provides endpoints for detecting potential scams in Web3 transactions using a Multi-Task Learning (MTL) model. The model can analyze both account-level and transaction-level patterns to identify suspicious activities.

## Features

- **Multi-Task Learning**: Simultaneously analyzes both account and transaction patterns
- **Real-time Prediction**: Fast inference for both account and transaction analysis
- **Feature Engineering**: Processes 15 most important features for each task
- **Docker Support**: Easy deployment using Docker containers
- **RESTful API**: Simple HTTP endpoints for integration

## Model Architecture

The API uses an MTL-MLP (Multi-Task Learning Multi-Layer Perceptron) model trained on Web3 transaction data with the following characteristics:

- **Input Features**: 15 carefully selected features for each task
- **Shared Layers**: 128-dimensional shared representation
- **Task-Specific Heads**: 64-dimensional task-specific layers
- **Output**: Probability scores for both account and transaction-level scam detection

## API Endpoints

### 1. Predict Endpoint

```http
POST /predict
```

Request body:
```json
{
    "account_address": "0x...",
    "transaction_history": [
        {
            "from_address": "0x...",
            "to_address": "0x...",
            "value": 1000000000000000000,
            "gasPrice": 20000000000,
            "gasUsed": 21000,
            "timestamp": 1636500000,
            "function_call": "[]",
            "token_value": 0,
            "nft_floor_price": 0,
            "nft_average_price": 0,
            "nft_total_volume": 0,
            "nft_total_sales": 0,
            "nft_num_owners": 0,
            "nft_market_cap": 0
        }
        // ... more transactions
    ]
}
```

Response:
```json
{
    "account_scam_probability": 0.123,
    "transaction_scam_probability": 0.456
}
```

### 2. Health Check

```http
GET /health
```

Response:
```json
{
    "status": "healthy"
}
```

## Setup and Deployment

### Prerequisites

- Windows 10/11 Pro, Enterprise, or Education (for Docker Desktop)
- At least 4GB RAM
- Python 3.9+ (for local development)
- WSL2 (Windows Subsystem for Linux 2)

### Docker Setup (Windows)

1. Install WSL2 (Windows Subsystem for Linux 2):
```powershell
# Run in PowerShell as Administrator
wsl --install
```
Restart your computer after installation.

2. Install Docker Desktop:
- Download Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- Run the installer
- During installation, ensure the "Use WSL 2 instead of Hyper-V" option is selected
- After installation, restart your computer

3. Verify Docker installation:
```powershell
docker --version
docker ps
```

If you see the error "error during connect: ... dockerDesktopLinuxEngine", try these steps:
- Open Docker Desktop application
- Wait for Docker Engine to start (check the whale icon in system tray)
- Ensure Docker Desktop is running with WSL 2 (Settings > General)
- Try restarting Docker Desktop

### Using Docker

1. Navigate to the api directory:
```powershell
cd Deploy/api
```

2. Build the Docker image:
```powershell
docker build -t web3-scam-detection .
```

3. Run the container:
```powershell
docker run -p 8000:8000 web3-scam-detection
```

The API will be available at `http://localhost:8000`

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Model Features

### Account-Level Features (in order of importance)
1. avg_gas_price
2. activity_duration_days
3. std_time_between_txns
4. total_volume
5. inNeighborNum
6. total_txn
7. in_out_ratio
8. total_value_in
9. outNeighborNum
10. avg_gas_used
11. giftinTxn_ratio
12. miningTxnNum
13. avg_value_out
14. turnover_ratio
15. out_txn

### Transaction-Level Features (in order of importance)
1. gas_price
2. gas_used
3. value
4. num_functions
5. has_suspicious_func
6. nft_num_owners
7. nft_total_sales
8. token_value
9. nft_total_volume
10. is_mint
11. high_gas
12. nft_average_price
13. nft_floor_price
14. nft_market_cap
15. is_zero_value

## API Documentation

After running the API, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
Deploy/
├── api/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model architecture
│   ├── data_processor.py    # Feature processing
│   ├── requirements.txt     # Dependencies
│   ├── Dockerfile          # Container configuration
│   ├── models/             # Directory for model files
│   │   └── MTL_MLP_best.pth
│   └── features/           # Directory for feature configuration
│       ├── AccountLevel_top15_features.json
│       └── TransactionLevel_top15_features.json
└── README.md              # This file
```

### Pre-deployment Setup

Before building the Docker image, copy the required files:
```powershell
# Create directories
mkdir -p api/models api/features

# Copy model and feature files
copy "../all_outputs/kaggle/working/MTL_MLP/MTL_MLP_best.pth" "api/models/"
copy "../all_outputs/kaggle/working/Feature_Importance/AccountLevel_top15_features.json" "api/features/"
copy "../all_outputs/kaggle/working/Feature_Importance/TransactionLevel_top15_features.json" "api/features/"
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid input data
- Missing model files
- Feature calculation errors
- Invalid transaction formats


