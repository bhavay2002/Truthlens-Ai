"""
Comprehensive test suite for TruthLens AI
"""
import pytest
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectStructure:
    """Test project structure and setup"""
    
    def test_project_structure_exists(self):
        """Test essential directories exist"""
        assert Path("src").exists()
        assert Path("config/config.yaml").exists()
        assert Path("api").exists()
        assert Path("models").exists()
        assert Path("data").exists()
    
    def test_source_modules_exist(self):
        """Test source code modules"""
        assert Path("src/data").exists()
        assert Path("src/models").exists()
        assert Path("src/features").exists()
        assert Path("src/evaluation").exists()
    
    def test_config_files_exist(self):
        """Test configuration files"""
        assert Path("config/config.yaml").exists()
        assert Path("requirements.txt").exists()
        assert Path(".env.example").exists()


class TestDataProcessing:
    """Test data processing functions"""
    
    def test_clean_text(self):
        """Test text cleaning function"""
        from src.data.clean_data import clean_text
        
        text = "Check this URL: https://example.com and some CAPS!"
        cleaned = clean_text(text)
        
        assert "https" not in cleaned
        assert cleaned.islower()
        assert cleaned.isalpha() or ' ' in cleaned
    
    def test_clean_dataframe(self):
        """Test dataframe cleaning"""
        from src.data.clean_data import clean_dataframe
        
        # Create sample data
        df = pd.DataFrame({
            'text': ['Sample news 1', 'Sample news 2', 'Sample news 1', None],
            'label': [0, 1, 0, 1]
        })
        
        cleaned = clean_dataframe(df)
        
        # Should remove duplicates and nulls
        assert len(cleaned) < len(df)
        assert cleaned['text'].notnull().all()


class TestFeatureEngineering:
    """Test feature engineering modules"""
    
    def test_tfidf_features(self):
        """Test TF-IDF feature extraction"""
        from src.features.text_features import tfidf_features
        
        texts = ["this is sample news", "another news article", "fake news story"]
        X, vectorizer = tfidf_features(texts)
        
        assert X.shape[0] == len(texts)
        assert vectorizer is not None
    
    def test_source_credibility(self):
        """Test source credibility scoring"""
        from src.features.source_features import source_credibility
        
        high_cred = source_credibility("bbc.com")
        low_cred = source_credibility("unknown.com")
        
        assert high_cred == 1
        assert low_cred == 0
    
    def test_metadata_features(self):
        """Test metadata feature extraction"""
        from src.features.metadata_features import extract_metadata_features
        
        df = pd.DataFrame({
            'title': ['News title', 'Another title'],
            'text': ['Some news text here', 'More news content'],
            'author': ['John Doe', None]
        })
        
        df_featured = extract_metadata_features(df)
        
        assert 'title_length' in df_featured.columns
        assert 'text_length' in df_featured.columns
        assert 'has_author' in df_featured.columns


class TestModelUtils:
    """Test model utility functions"""
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        from src.models.model_utils import save_model, load_model
        import tempfile
        
        # Create dummy model
        dummy_model = {"type": "test", "params": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Save and load
            save_model(dummy_model, tmp_path)
            loaded_model = load_model(tmp_path)
            
            assert loaded_model == dummy_model
        finally:
            tmp_file = Path(tmp_path)
            if tmp_file.exists():
                tmp_file.unlink()


class TestEvaluation:
    """Test evaluation functions"""
    
    def test_evaluate_function(self):
        """Test model evaluation metrics"""
        from src.evaluation.evaluate_model import evaluate
        
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 1, 0, 0]
        
        results = evaluate(y_true, y_pred)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 0 <= results['accuracy'] <= 1


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api.app import app
        return TestClient(app)
    
    def test_home_endpoint(self, client):
        """Test home endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint input validation"""
        # Test with invalid input (too short)
        response = client.post(
            "/predict",
            json={"text": "short"}
        )
        assert response.status_code == 422  # Validation error


class TestDataValidation:
    """Test data validation utilities"""
    
    def test_validator_schema(self):
        """Test schema validation"""
        from src.data.validate_data import DataValidator
        
        validator = DataValidator(required_columns=['text', 'label'])
        
        # Valid dataframe
        df_valid = pd.DataFrame({'text': ['sample'], 'label': [0]})
        assert validator.validate_schema(df_valid)
        
        # Invalid dataframe
        df_invalid = pd.DataFrame({'text': ['sample']})
        assert not validator.validate_schema(df_invalid)
    
    def test_validator_nulls(self):
        """Test null value validation"""
        from src.data.validate_data import DataValidator
        
        validator = DataValidator()
        
        df = pd.DataFrame({
            'text': ['sample', None, 'text'],
            'label': [0, 1, 0]
        })
        
        result = validator.validate_nulls(df, max_null_ratio=0.2)
        assert not result  # Should fail due to nulls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

