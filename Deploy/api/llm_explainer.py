import google.generativeai as genai
from typing import Dict, List
import os

class LLMExplainer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    def _create_explanation_prompt(self, prediction_results: Dict, shap_values: Dict) -> str:
        """Tạo prompt chi tiết cho Gemini"""
        account_prob = prediction_results["account_scam_probability"]
        transaction_prob = prediction_results["transaction_scam_probability"]
        
        account_features = shap_values["account"]["feature_importance"]
        transaction_features = shap_values["transaction"]["feature_importance"]
        
        prompt = f"""You are a Web3 security expert. Explain the following blockchain transaction analysis results in simple terms that a non-technical person can understand.

PREDICTION RESULTS:
- Account Scam Probability: {account_prob:.2%}
- Transaction Scam Probability: {transaction_prob:.2%}

TOP INFLUENTIAL FACTORS FOR ACCOUNT ANALYSIS:
{self._format_features_for_prompt(account_features[:5])}

TOP INFLUENTIAL FACTORS FOR TRANSACTION ANALYSIS:
{self._format_features_for_prompt(transaction_features[:5])}

Please provide:
1. An overall risk assessment in simple terms
2. The main suspicious patterns detected (if any)
3. Recommendations for the user
4. A simple explanation of why the model made this decision

Keep the explanation conversational and easy to understand. Avoid technical jargon where possible, and when using technical terms, explain them simply.
Focus on practical implications rather than technical details."""

        return prompt
    
    def _format_features_for_prompt(self, features: List[Dict]) -> str:
        """Format feature importance data for the prompt"""
        formatted = []
        for f in features:
            impact = "increasing risk" if f["shap_value"] > 0 else "decreasing risk"
            formatted.append(f"- {f['feature_name']}: value = {f['feature_value']:.2f} ({impact})")
        return "\n".join(formatted)
    
    def _translate_feature_name(self, name: str) -> str:
        """Translate technical feature names to human-readable descriptions"""
        translations = {
            "avg_gas_price": "average transaction fee",
            "activity_duration_days": "account age in days",
            "std_time_between_txns": "irregularity in transaction timing",
            "total_volume": "total amount transferred",
            "inNeighborNum": "number of unique senders",
            "total_txn": "total number of transactions",
            "in_out_ratio": "ratio of incoming to outgoing transactions",
            "total_value_in": "total amount received",
            "outNeighborNum": "number of unique recipients",
            "avg_gas_used": "average transaction complexity",
            "giftinTxn_ratio": "proportion of token transfers",
            "miningTxnNum": "number of mining transactions",
            "avg_value_out": "average amount sent",
            "turnover_ratio": "frequency of fund movements",
            "out_txn": "number of outgoing transactions",
            "gas_price": "transaction fee",
            "gas_used": "transaction complexity",
            "value": "transaction amount",
            "num_functions": "number of contract interactions",
            "has_suspicious_func": "presence of suspicious functions",
            "nft_num_owners": "number of NFT owners",
            "nft_total_sales": "total NFT sales volume",
            "token_value": "token transfer value",
            "nft_total_volume": "total NFT trading volume",
            "is_mint": "is a new token creation",
            "high_gas": "high transaction fee",
            "nft_average_price": "average NFT price",
            "nft_floor_price": "minimum NFT price",
            "nft_market_cap": "total NFT market value",
            "is_zero_value": "zero-value transaction"
        }
        return translations.get(name, name)

    async def get_user_friendly_explanation(self, prediction_results: Dict, shap_values: Dict) -> str:
        """
        Tạo giải thích dễ hiểu bằng cách sử dụng Gemini
        """
        try:
            prompt = self._create_explanation_prompt(prediction_results, shap_values)
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"""I apologize, but I couldn't generate a detailed explanation at the moment.

Quick Summary:
- Account Risk: {prediction_results['account_scam_probability']:.2%}
- Transaction Risk: {prediction_results['transaction_scam_probability']:.2%}

Please review the raw SHAP values for detailed insights.
Error: {str(e)}"""