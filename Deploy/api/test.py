import requests
import json
import os
from dotenv import load_dotenv

def test_llm_predictions():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY not found in .env file")
        return

    BASE_URL = "http://localhost:8000"

    # Test data with both legitimate and suspicious patterns
    test_cases = [
        {
            "name": "Legitimate Transaction",
            "data": {
                "account_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "transaction_history": [{
                    "from_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                    "to_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "value": 1000000000000000000,
                    "gasPrice": 20000000000,
                    "gasUsed": 21000,
                    "timestamp": 1636500000,
                    "function_call": "[]",
                    "token_value": 0,
                    "nft_floor_price": 100000000000000000,
                    "nft_average_price": 150000000000000000,
                    "nft_total_volume": 1000000000000000000000,
                    "nft_total_sales": 1000,
                    "nft_num_owners": 500,
                    "nft_market_cap": 5000000000000000000000
                }],
                "explain": False,
                "explain_with_llm": False
            }
        },
        {
            "name": "Suspicious Transaction",
            "data": {
                "account_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "transaction_history": [{
                    "from_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                    "to_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "value": 0,  # Zero value transfer
                    "gasPrice": 50000000000,  # High gas price
                    "gasUsed": 150000,  # High gas usage
                    "timestamp": 1636500000,
                    "function_call": "[\"setApprovalForAll\"]",  # Suspicious function
                    "token_value": 1000000000000000000,
                    "nft_floor_price": 1000000000000000000,
                    "nft_average_price": 900000000000000000,  # Below floor price
                    "nft_total_volume": 100000000000000000000,
                    "nft_total_sales": 50,
                    "nft_num_owners": 10,  # Low number of owners
                    "nft_market_cap": 500000000000000000000
                }],
                "explain": False,
                "explain_with_llm": False
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print("-" * 80)
        
        try:
            # Make prediction request
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful
            response.raise_for_status()
            result = response.json()
            
            # Print formatted results
            print("\n📊 Prediction Results:")
            print(f"Account Scam Probability: {result['account_scam_probability']:.4f}")
            print(f"Transaction Scam Probability: {result['transaction_scam_probability']:.4f}")
            
            # Print SHAP explanations if available
            if "shap_values" in result:
                print("\n🔍 SHAP Explanations:")
                print("Account Level Top Features:")
                for feature in result["shap_values"]["account"]["feature_importance"][:3]:
                    print(f"  - {feature['feature']}: {feature['importance']:.4f}")
                    
                print("\nTransaction Level Top Features:")
                for feature in result["shap_values"]["transaction"]["feature_importance"][:3]:
                    print(f"  - {feature['feature']}: {feature['importance']:.4f}")
            
            # Print LLM explanation if available
            if "llm_explanation" in result:
                print("\n🤖 LLM Explanation:")
                print(result["llm_explanation"])
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making request: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            
        print("-" * 80)

if __name__ == "__main__":
    test_llm_predictions()