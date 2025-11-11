from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import numpy as np
import os
from model import MTL_MLP
from data_processor import process_account_data, process_transaction_data
from dotenv import load_dotenv
load_dotenv()
# Initialize LLM Explainer with Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_explainer = None
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. LLM explanations will not be available.")
else:
    try:
        from llm_explainer import LLMExplainer
        llm_explainer = LLMExplainer(GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: LLM explainer could not be initialized: {e}")

app = FastAPI(title="Web3 Scam Detection API",
             description="API for detecting scam accounts and transactions using MTL-MLP model with SHAP and LLM explanations",
             version="1.0.0")

# Define paths (use deploy api folder as base)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

# Load model
MODEL_PATH = os.path.join(MODEL_DIR, 'MTL_MLP_best.pth')

# Initialize model with correct architecture
model = MTL_MLP(
    input_dim=15,
    shared_dim=128,
    head_hidden_dim=64
)

# Load model weights robustly
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# support checkpoints that wrap state dict
if isinstance(ckpt, dict):
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
else:
    state = ckpt

# strip module prefix if present
new_state = {}
for k, v in state.items():
    new_k = k.replace('module.', '') if k.startswith('module.') else k
    new_state[new_k] = v

model.load_state_dict(new_state)
model.eval()

# Load feature lists
account_features_path = os.path.join(FEATURES_DIR, "AccountLevel_top15_features.json")
transaction_features_path = os.path.join(FEATURES_DIR, "TransactionLevel_top15_features.json")

if not os.path.exists(account_features_path) or not os.path.exists(transaction_features_path):
    raise FileNotFoundError("Feature importance files not found")

with open(account_features_path, "r") as f:
    ACCOUNT_FEATURES = json.load(f)

with open(transaction_features_path, "r") as f:
    TRANSACTION_FEATURES = json.load(f)


def _feature_name_list(feature_json):
    # Accept either list of strings or list of dicts with 'feature' key
    if not feature_json:
        return []
    if isinstance(feature_json[0], str):
        return feature_json
    elif isinstance(feature_json[0], dict) and 'feature' in feature_json[0]:
        return [f['feature'] for f in feature_json]
    else:
        # fallback: convert items to str
        return [str(f) for f in feature_json]

ACCOUNT_FEATURE_NAMES = _feature_name_list(ACCOUNT_FEATURES)
TRANSACTION_FEATURE_NAMES = _feature_name_list(TRANSACTION_FEATURES)

class AccountRequest(BaseModel):
    account_address: str
    transaction_history: list
    explain: bool = False  # Optional parameter to request SHAP explanation
    explain_with_llm: bool = False  # Optional parameter to request LLM explanation

@app.post("/predict")
async def predict_scam(request: AccountRequest):
    try:
        # Process account data
        account_features = process_account_data(request.account_address, request.transaction_history)
        
        # Process transaction data
        transaction_features = process_transaction_data(request.transaction_history)
        
        # Combine features
        features = torch.tensor(
            np.concatenate([account_features, transaction_features]), 
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Make prediction: model expects 15-d input per task
        with torch.no_grad():
            # account features -> account prediction (logit)
            acct_logit = model(features[:, :15], task_id='account').squeeze()
            txn_logit = model(features[:, 15:], task_id='transaction').squeeze()

            account_prob = torch.sigmoid(acct_logit)
            transaction_prob = torch.sigmoid(txn_logit)

        response = {
            "account_scam_probability": float(account_prob.item()),
            "transaction_scam_probability": float(transaction_prob.item())
        }
        
        # Add SHAP explanations if requested
        shap_explanations = None
        if request.explain or request.explain_with_llm:
            # Import SHAP explainer lazily so app can run without SHAP installed
            try:
                from shap_explainer import SHAPExplainer
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"SHAP is not available: {e}")

            # Initialize SHAP explainer if not already done
            if not hasattr(app.state, 'shap_explainer'):
                # initialize explainer with CPU device
                app.state.shap_explainer = SHAPExplainer(model, device='cpu')

            # Get SHAP values for account prediction
            account_explanation = app.state.shap_explainer.explain_prediction(
                features[:, :15].numpy(),  # First 15 features for account
                'account',
                ACCOUNT_FEATURE_NAMES
            )

            # Get SHAP values for transaction prediction
            transaction_explanation = app.state.shap_explainer.explain_prediction(
                features[:, 15:].numpy(),  # Last 15 features for transaction
                'transaction',
                TRANSACTION_FEATURE_NAMES
            )

            shap_explanations = {
                "account": account_explanation,
                "transaction": transaction_explanation
            }
            response["explanations"] = shap_explanations

        # Add LLM explanation if requested
        if request.explain_with_llm:
            if llm_explainer is None:
                raise HTTPException(
                    status_code=501,
                    detail="LLM explanations are not available. Please set GEMINI_API_KEY environment variable."
                )
            
            llm_explanation = await llm_explainer.get_user_friendly_explanation(
                prediction_results=response,
                shap_values=shap_explanations
            )
            response["llm_explanation"] = llm_explanation
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}