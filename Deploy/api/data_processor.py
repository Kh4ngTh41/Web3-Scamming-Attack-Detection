import numpy as np
from typing import List, Dict
import json

def process_account_data(account_address: str, transactions: List[Dict]) -> np.ndarray:
    """
    Process account-level features from transaction history.
    Returns the top 15 important features for account-level prediction.
    """
    features = {}
    
    # Convert account address to lowercase
    account_address = account_address.lower()
    
    # Calculate features according to importance ranking
    
    # 1. Average gas price
    gas_prices = [tx.get('gasPrice', 0) for tx in transactions]
    features['avg_gas_price'] = np.mean(gas_prices) if gas_prices else 0
    
    # 2. Activity duration in days
    timestamps = [tx.get('timestamp', 0) for tx in transactions]
    if timestamps:
        activity_duration_days = (max(timestamps) - min(timestamps)) / (24 * 3600)
        features['activity_duration_days'] = activity_duration_days
    else:
        features['activity_duration_days'] = 0
        
    # 3. Standard deviation of time between transactions
    if len(timestamps) > 1:
        sorted_timestamps = sorted(timestamps)
        time_diffs = np.diff(sorted_timestamps)
        features['std_time_between_txns'] = np.std(time_diffs)
    else:
        features['std_time_between_txns'] = 0
        
    # 4. Total volume (sum of all transaction values)
    values = [tx.get('value', 0) for tx in transactions]
    features['total_volume'] = sum(values)
    
    # 5. Number of unique incoming neighbors
    in_neighbors = set(tx['from_address'] for tx in transactions if tx['to_address'].lower() == account_address)
    features['inNeighborNum'] = len(in_neighbors)
    
    # 6. Total number of transactions
    features['total_txn'] = len(transactions)
    
    # 7. In/Out transaction ratio
    out_txns = [tx for tx in transactions if tx['from_address'].lower() == account_address]
    in_txns = [tx for tx in transactions if tx['to_address'].lower() == account_address]
    features['in_out_ratio'] = len(in_txns) / max(len(out_txns), 1)
    
    # 8. Total value received
    features['total_value_in'] = sum(tx.get('value', 0) for tx in in_txns)
    
    # 9. Number of unique outgoing neighbors
    out_neighbors = set(tx['to_address'] for tx in transactions if tx['from_address'].lower() == account_address)
    features['outNeighborNum'] = len(out_neighbors)
    
    # 10. Average gas used
    gas_used = [tx.get('gasUsed', 0) for tx in transactions]
    features['avg_gas_used'] = np.mean(gas_used) if gas_used else 0
    
    # 11. Ratio of gift-in transactions
    gift_in_txns = [tx for tx in in_txns if tx.get('value', 0) == 0 and tx.get('token_value', 0) > 0]
    features['giftinTxn_ratio'] = len(gift_in_txns) / max(len(in_txns), 1)
    
    # 12. Number of mining transactions
    mining_txns = [tx for tx in transactions if tx.get('from_address', '').startswith('0x0000000000000000000000000000000000000000')]
    features['miningTxnNum'] = len(mining_txns)
    
    # 13. Average value of outgoing transactions
    features['avg_value_out'] = np.mean([tx.get('value', 0) for tx in out_txns]) if out_txns else 0
    
    # 14. Turnover ratio
    features['turnover_ratio'] = len(out_txns) / max(len(in_txns), 1)
    
    # 15. Number of outgoing transactions
    features['out_txn'] = len(out_txns)

    # Return features in the order specified by AccountLevel_top15_features.json
    ordered_features = [
        'avg_gas_price', 'activity_duration_days', 'std_time_between_txns',
        'total_volume', 'inNeighborNum', 'total_txn', 'in_out_ratio',
        'total_value_in', 'outNeighborNum', 'avg_gas_used', 'giftinTxn_ratio',
        'miningTxnNum', 'avg_value_out', 'turnover_ratio', 'out_txn'
    ]
    
    return np.array([features[f] for f in ordered_features])

def process_transaction_data(transactions: List[Dict]) -> np.ndarray:
    """
    Process transaction-level features.
    Returns the top 15 important features for transaction-level prediction.
    """
    if not transactions:
        return np.zeros(15)
    
    features = {}
    
    # 1. Gas price
    features['gas_price'] = np.mean([tx.get('gasPrice', 0) for tx in transactions])
    
    # 2. Gas used
    features['gas_used'] = np.mean([tx.get('gasUsed', 0) for tx in transactions])
    
    # 3. Transaction value
    features['value'] = np.mean([tx.get('value', 0) for tx in transactions])
    
    # 4. Number of functions called
    features['num_functions'] = np.mean([
        len(tx.get('function_list', [])) if isinstance(tx.get('function_list'), list)
        else (len(json.loads(tx.get('function_call', '[]'))) if tx.get('function_call') else 0)
        for tx in transactions
    ])
    
    # 5. Has suspicious functions
    suspicious_patterns = ['setApprovalForAll', 'approve', 'transferFrom', 'safeTransferFrom',
                         'batchTransfer', 'multiTransfer', 'permit', 'delegateCall']
    
    def has_suspicious_func(tx):
        funcs = tx.get('function_list', []) if isinstance(tx.get('function_list'), list) else \
                json.loads(tx.get('function_call', '[]')) if tx.get('function_call') else []
        return any(pattern.lower() in func.lower() for func in funcs for pattern in suspicious_patterns)
    
    features['has_suspicious_func'] = np.mean([1 if has_suspicious_func(tx) else 0 for tx in transactions])
    
    # 6-15. NFT and Token related features
    features['nft_num_owners'] = np.mean([tx.get('nft_num_owners', 0) for tx in transactions])
    features['nft_total_sales'] = np.mean([tx.get('nft_total_sales', 0) for tx in transactions])
    features['token_value'] = np.mean([tx.get('token_value', 0) for tx in transactions])
    features['nft_total_volume'] = np.mean([tx.get('nft_total_volume', 0) for tx in transactions])
    
    # Mint detection
    features['is_mint'] = np.mean([
        1 if tx.get('from_address', '').startswith('0x0000000000000000000000000000000000000000') else 0 
        for tx in transactions
    ])
    
    # High gas detection (75th percentile threshold)
    gas_75th = np.percentile([tx.get('gasUsed', 0) for tx in transactions], 75)
    features['high_gas'] = np.mean([1 if tx.get('gasUsed', 0) > gas_75th else 0 for tx in transactions])
    
    features['nft_average_price'] = np.mean([tx.get('nft_average_price', 0) for tx in transactions])
    features['nft_floor_price'] = np.mean([tx.get('nft_floor_price', 0) for tx in transactions])
    features['nft_market_cap'] = np.mean([tx.get('nft_market_cap', 0) for tx in transactions])
    features['is_zero_value'] = np.mean([1 if tx.get('value', 0) == 0 else 0 for tx in transactions])

    # Return features in the order specified by TransactionLevel_top15_features.json
    ordered_features = [
        'gas_price', 'gas_used', 'value', 'num_functions', 'has_suspicious_func',
        'nft_num_owners', 'nft_total_sales', 'token_value', 'nft_total_volume',
        'is_mint', 'high_gas', 'nft_average_price', 'nft_floor_price',
        'nft_market_cap', 'is_zero_value'
    ]
    
    return np.array([features[f] for f in ordered_features])