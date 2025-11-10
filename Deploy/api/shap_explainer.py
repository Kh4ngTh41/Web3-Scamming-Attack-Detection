import torch
import numpy as np
import shap
from model import SHAPModelWrapper

class SHAPExplainer:
    def __init__(self, model, background_data_size=100):
        self.model = model
        self.background_data_size = background_data_size
        self.background_data = None
        self.account_explainer = None
        self.transaction_explainer = None
        
    def prepare_background_data(self, sample_features):
        """
        Chuẩn bị background data cho SHAP từ một mẫu features
        """
        # Tạo background data ngẫu nhiên dựa trên phân phối của mẫu
        mean = np.mean(sample_features, axis=0)
        std = np.std(sample_features, axis=0)
        self.background_data = np.random.normal(
            mean, std, 
            (self.background_data_size, sample_features.shape[1])
        )
        self.background_tensor = torch.FloatTensor(self.background_data)
        
    def explain_prediction(self, features, task_id, feature_names):
        """
        Giải thích dự đoán cho một mẫu cụ thể
        """
        # Kiểm tra và chuẩn bị background data nếu cần
        if self.background_data is None:
            self.prepare_background_data(features)
            
        # Wrap model cho task cụ thể
        wrapped_model = SHAPModelWrapper(self.model, task_id)
        
        # Khởi tạo DeepExplainer
        if task_id == 'account':
            if self.account_explainer is None:
                self.account_explainer = shap.DeepExplainer(
                    wrapped_model, 
                    self.background_tensor
                )
            explainer = self.account_explainer
        else:  # transaction
            if self.transaction_explainer is None:
                self.transaction_explainer = shap.DeepExplainer(
                    wrapped_model, 
                    self.background_tensor
                )
            explainer = self.transaction_explainer
            
        # Chuyển features sang tensor
        features_tensor = torch.FloatTensor(features)
        
        # Tính SHAP values
        shap_values = explainer.shap_values(features_tensor)
        
        # Chuyển đổi SHAP values thành định dạng dễ hiểu
        feature_importance = []
        for idx, (name, value) in enumerate(zip(feature_names, shap_values[0])):
            feature_importance.append({
                "feature_name": name,
                "shap_value": float(value),
                "feature_value": float(features[0][idx])
            })
            
        # Sắp xếp theo giá trị tuyệt đối của SHAP value
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "base_value": float(explainer.expected_value),
            "feature_importance": feature_importance
        }