"""Integration test script for the backend API and local model.

What it does:
- Loads or generates 10 sample transactions
- Calls `process_account_data` to build account features
- Creates an 11th transaction (new) and appends to history
- Loads the MTL_MLP model weights (robustly) and runs local predictions for account and tx11
- Sends a POST request to the running API `/predict` endpoint and prints the response

Usage:
  Activate your venv and run:
    python run_multi_task_test.py

Adjust API_URL at top if your server runs elsewhere.
"""

import os
import json
import time
import random
import pprint

import numpy as np

try:
    import requests
except Exception:
    requests = None

import torch

from data_processor import process_account_data, process_transaction_data
from model import MTL_MLP


API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000/predict')
BASE = os.path.dirname(__file__)
MODEL_PATHS = [
    os.path.join(BASE, 'models', 'MTL_MLP_best.pth'),
    r'F:\DATA\Web 3 Scamming\Deploy\api\models\MTL_MLP_best.pth'
]


def load_or_init_model(input_dim=15):
    model = MTL_MLP(input_dim=input_dim)
    # try multiple candidate paths
    ckpt_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path:
        print(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            else:
                state = ckpt
        else:
            state = ckpt
        # strip module prefix
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '') if k.startswith('module.') else k] = v
        model.load_state_dict(new_state)
        print("Model loaded.")
    else:
        print("No model weights found; using randomly initialized model.")
    model.eval()
    return model


def generate_transactions(n=10):
    sample_file = os.path.join(BASE, 'test_data_sample.json')
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            txns = json.load(f)
        # If file contains >= n, take first n
        if len(txns) >= n:
            return txns[:n]

    # Otherwise generate synthetic transactions
    txns = []
    now = int(time.time())
    for i in range(n):
        tx = {
            'from_address': f'0xfrom{i:040x}'[-42:],
            'to_address': f'0xto{i:040x}'[-42:],
            'value': random.randint(0, 5) * 10**18,
            'gasPrice': random.randint(10, 200) * 10**9,
            'gasUsed': random.randint(21000, 200000),
            'timestamp': now - (n - i) * 3600,
            'function_call': '[]',
            'token_value': 0,
            'nft_floor_price': 0,
            'nft_average_price': 0,
            'nft_total_volume': 0,
            'nft_total_sales': 0,
            'nft_num_owners': 0,
            'nft_market_cap': 0,
            'tx_type': 'transfer',
            'contract_address': '0x0000000000000000000000000000000000000000',
            'transaction_hash': f'0xhash{i}'
        }
        txns.append(tx)
    return txns


def make_new_transaction(i=11):
    now = int(time.time())
    return {
        'from_address': '0xsuspicious000000000000000000000000000000',
        'to_address': '0xvictim0000000000000000000000000000000000',
        'value': 3 * 10**18,
        'gasPrice': 250 * 10**9,
        'gasUsed': 150000,
        'timestamp': now,
        'function_call': '["transferFrom"]',
        'token_value': 0,
        'nft_floor_price': 0,
        'nft_average_price': 0,
        'nft_total_volume': 0,
        'nft_total_sales': 0,
        'nft_num_owners': 0,
        'nft_market_cap': 0,
        'tx_type': 'erc721',
        'contract_address': '0xsuspiciouscontract000000000000000000000',
        'transaction_hash': f'0xnewhash{i}'
    }


def run_test():
    print("Generating transactions...")
    txns = generate_transactions(10)
    print(f"Generated {len(txns)} transactions")

    # pick account address to analyze (use from_address of tx0)
    account = txns[0]['from_address']

    # process account features using provided preprocessor
    acct_feats = process_account_data(account, txns)
    print("Account features vector (len={}):".format(len(acct_feats)))
    print(acct_feats)

    # create new tx (11th)
    tx11 = make_new_transaction()
    print("New transaction (11):")
    pprint.pprint(tx11)

    # evaluate locally using model
    model = load_or_init_model(input_dim=15)

    # Prepare features for model prediction
    # account features -> account input
    acct_input = torch.tensor(acct_feats, dtype=torch.float32).unsqueeze(0)
    # transaction features: take features from tx11 via process_transaction_data
    txn11_feats = process_transaction_data([tx11])
    txn_input = torch.tensor(txn11_feats, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        acct_logit = model(acct_input, task_id='account').squeeze()
        txn_logit = model(txn_input, task_id='transaction').squeeze()
        acct_prob = torch.sigmoid(acct_logit).item()
        txn_prob = torch.sigmoid(txn_logit).item()

    print(f"Local model prediction -> account_prob={acct_prob:.6f}, txn11_prob={txn_prob:.6f}")

    # Call API endpoint
    payload = {
        'account_address': account,
        'transaction_history': txns + [tx11],
        'explain': True,
        'explain_with_llm': False
    }

    if requests is None:
        print("requests package not installed; skipping API call. To enable, pip install requests")
        return

    print(f"Sending request to API at {API_URL} ...")
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        print('Status code:', r.status_code)
        try:
            print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(r.text)
    except Exception as e:
        print('Failed to call API:', e)


if __name__ == '__main__':
    run_test()
"""
Script: run_multi_task_test.py

Mục tiêu:
- Sinh ra 10 giao dịch mẫu, dùng `process_account_data` để tạo feature cho account
- Sinh thêm giao dịch thứ 11 (một giao dịch nghi ngờ) và tính feature transaction-level cho nó
- Load model `models/MTL_MLP_best.pth` nếu có, hoặc dùng model khởi tạo ngẫu nhiên
- Chạy dự đoán cho task account và task transaction (giao dịch thứ 11)
- In ra feature vectors, xác suất, và lưu kết quả vào `multi_task_test_results.json`

Chạy: python run_multi_task_test.py
"""

import os
import json
import numpy as np
import torch
from data_processor import process_account_data, process_transaction_data
from model import MTL_MLP


def generate_ten_transactions(base_time=1700000000):
    txns = []
    for i in range(10):
        tx = {
            "from_address": f"0xfrom{i:02d}abcd000000000000000000000000000000{i}",
            "to_address": "0xMY_TEST_ACCOUNT_ADDRESS",
            "value": int((i + 1) * 1e17),
            "gasPrice": int(20e9 + i * 1e9),
            "gasUsed": 21000 + i * 1000,
            "timestamp": base_time + i * 3600,
            "function_call": "[]",
            "token_value": 0,
            "nft_floor_price": 0,
            "nft_average_price": 0,
            "nft_total_volume": 0,
            "nft_total_sales": 0,
            "nft_num_owners": 0,
            "nft_market_cap": 0,
            "tx_type": "transfer",
            "contract_address": "0x0000000000000000000000000000000000000000",
            "transaction_hash": f"0xtxn{i:02d}"
        }
        txns.append(tx)
    return txns


def generate_suspicious_transaction(base_time=1700000000):
    # Giao dịch thứ 11: high gas, suspicious function, zero value with token transfer
    return {
        "from_address": "0xsuspicious_actor000000000000000000000000",
        "to_address": "0xMY_TEST_ACCOUNT_ADDRESS",
        "value": 0,
        "gasPrice": int(120e9),
        "gasUsed": 250000,
        "timestamp": base_time + 11 * 3600,
        "function_call": "[\"setApprovalForAll\", \"approve\"]",
        "token_value": int(5e18),
        "nft_floor_price": int(1e18),
        "nft_average_price": int(9e17),
        "nft_total_volume": int(1e21),
        "nft_total_sales": 50,
        "nft_num_owners": 5,
        "nft_market_cap": int(5e20),
        "tx_type": "erc721",
        "contract_address": "0xsuspicious_contract0000000000000000000",
        "transaction_hash": "0xtxn11_suspicious"
    }


def load_model(model_path):
    model = MTL_MLP(input_dim=15)
    if not os.path.exists(model_path):
        print(f"[WARN] Không tìm thấy model tại {model_path}, sẽ dùng model khởi tạo ngẫu nhiên.")
        return model

    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # strip potential 'module.' prefix
    new_state = {}
    try:
        for k, v in state.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            new_state[new_k] = v
        model.load_state_dict(new_state)
        print(f"[OK] Loaded model weights from {model_path}")
    except Exception as e:
        print(f"[ERROR] Không thể load state_dict từ {model_path}: {e}\nSử dụng model khởi tạo ngẫu nhiên.")

    return model


def main():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, 'models', 'MTL_MLP_best.pth')

    # 1) Tạo 10 giao dịch
    txns = generate_ten_transactions()
    print(f"Tạo {len(txns)} giao dịch mẫu cho account test.")

    # 2) Tạo feature account bằng process_account_data
    account_addr = '0xMY_TEST_ACCOUNT_ADDRESS'
    acct_features = process_account_data(account_addr, txns)
    print("Account features (15):")
    print(acct_features.tolist())

    # 3) Tạo giao dịch thứ 11
    tx11 = generate_suspicious_transaction()
    print("\nGiao dịch thứ 11 (suspicious):")
    print(tx11)

    # 4) Tính feature cho giao dịch thứ 11
    txn11_feat = process_transaction_data([tx11])
    print("Transaction-11 features (15):")
    print(txn11_feat.tolist())

    # 5) Load model và dự đoán
    model = load_model(model_path)
    model.eval()

    with torch.no_grad():
        acct_tensor = torch.tensor(acct_features.reshape(1, -1), dtype=torch.float32)
        txn_tensor = torch.tensor(txn11_feat.reshape(1, -1), dtype=torch.float32)

        acct_logit = model(acct_tensor, task_id='account').squeeze()
        txn_logit = model(txn_tensor, task_id='transaction').squeeze()

        acct_prob = torch.sigmoid(acct_logit).item()
        txn_prob = torch.sigmoid(txn_logit).item()

    print(f"\nKết quả dự đoán:")
    print(f" - Xác suất tài khoản gian lận: {acct_prob:.6f}")
    print(f" - Xác suất giao dịch(11) gian lận: {txn_prob:.6f}")

    # 6) Lưu kết quả
    out = {
        'account_address': account_addr,
        'acct_features': acct_features.tolist(),
        'txn11': tx11,
        'txn11_features': txn11_feat.tolist(),
        'acct_prob': acct_prob,
        'txn11_prob': txn_prob
    }
    out_path = os.path.join(base, 'multi_task_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nĐã lưu kết quả vào: {out_path}")


if __name__ == '__main__':
    main()
