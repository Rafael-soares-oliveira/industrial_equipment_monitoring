import numpy as np
import pandas as pd
import pytest

from industrial_equipment_monitoring.pipelines.data_processing.nodes import (
    _prepare_cv_data,
    _split_data,
    data_processing,
    load_clean_data,
)


class TestDataProcessing:
    """Testes para as funções de processamento de dados"""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para testes"""
        np.random.seed(42)

        # Criar dados simulando equipamentos industriais
        n_samples = 100
        n_groups = 10

        data = {
            "equipment_id": np.random.choice(
                [f"EQ_{i}" for i in range(n_groups)], n_samples
            ),
            "vibration_x": np.random.normal(0, 1, n_samples),
            "vibration_y": np.random.normal(0, 1, n_samples),
            "temperature": np.random.normal(75, 10, n_samples),
            "pressure": np.random.normal(100, 20, n_samples),
            "rpm": np.random.normal(1500, 100, n_samples),
            "unused": np.random.normal(1500, 100, n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }

        # Adicionar alguns valores nulos e duplicatas
        df = pd.DataFrame(data)
        df.iloc[5:8, 2] = np.nan  # Adicionar nulos em vibration_y
        df = pd.concat([df, df.iloc[:3]])  # Adicionar duplicatas

        return df

    @pytest.fixture
    def parameters(self):
        """Parâmetros de configuração para testes"""
        return {
            "target_column": "target",
            "group_column": "equipment_id",
            "features_to_drop": ["unused"],
            "test_size": 0.2,
            "random_state": 42,
            "use_cross_validation": False,
        }

    def test_load_clean_data_basic(self, sample_data, parameters):
        """Testa a função básica de carregamento e limpeza"""
        # Act
        cleaned_data = load_clean_data(sample_data, parameters)

        # Assert
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0
        assert cleaned_data.isnull().sum().sum() == 0
        assert not cleaned_data.duplicated().any()

    def test_load_clean_data_removes_duplicates(self, sample_data, parameters):
        """Testa se duplicatas são removidas corretamente"""
        # Arrange
        initial_duplicates = sample_data.duplicated().sum()

        # Act
        cleaned_data = load_clean_data(sample_data, parameters)
        final_duplicates = cleaned_data.duplicated().sum()

        # Assert
        assert initial_duplicates > 0
        assert final_duplicates == 0

    def test_load_clean_data_removes_nulls(self, sample_data, parameters):
        """Testa se valores nulos são removidos corretamente"""
        # Arrange
        initial_nulls = sample_data.isnull().sum().sum()

        # Act
        cleaned_data = load_clean_data(sample_data, parameters)
        final_nulls = cleaned_data.isnull().sum().sum()

        # Assert
        assert initial_nulls > 0
        assert final_nulls == 0

    def test_load_clean_data_target_distribution(self, sample_data, parameters):
        """Testa o log da distribuição da target"""
        # Act
        cleaned_data = load_clean_data(sample_data, parameters)

        # Assert
        assert parameters["target_column"] in cleaned_data.columns
        target_values = cleaned_data[parameters["target_column"]].unique()
        assert set(target_values).issubset({0, 1})

    def test_split_data_basic(self, sample_data, parameters):
        """Testa a divisão básica dos dados"""
        # Arrange
        cleaned_data = load_clean_data(sample_data, parameters)

        # Act
        train_data, test_data = _split_data(cleaned_data, parameters)

        # Assert
        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert len(train_data) + len(test_data) == len(cleaned_data)

    def test_split_data_stratification(self, sample_data, parameters):
        """Testa se a estratificação está funcionando"""
        # Arrange
        cleaned_data = load_clean_data(sample_data, parameters)
        target_col = parameters["target_column"]
        train_p = 0.1
        test_p = 0.1

        # Act
        train_data, test_data = _split_data(cleaned_data, parameters)

        # Assert - verifica proporção similar entre treino e teste
        train_positive_ratio = train_data[target_col].mean()
        test_positive_ratio = test_data[target_col].mean()
        overall_ratio = cleaned_data[target_col].mean()

        # As proporções devem ser similares (dentro de uma tolerância)
        assert abs(train_positive_ratio - overall_ratio) < train_p
        assert abs(test_positive_ratio - overall_ratio) < test_p

    def test_prepare_cv_data_basic(self, sample_data, parameters):
        """Testa o preparo dos dados para validação cruzada"""
        # Arrange
        cleaned_data = load_clean_data(sample_data, parameters)

        # Act
        cv_data = _prepare_cv_data(cleaned_data, parameters)

        # Assert
        assert isinstance(cv_data, pd.DataFrame)
        assert parameters["target_column"] in cv_data.columns
        assert parameters["group_column"] in cv_data.columns
        assert "unused_feature" not in cv_data.columns

    def test_data_processing_cv_mode(self, sample_data, parameters):
        """Testa data_processing no modo validação cruzada"""
        # Arrange
        parameters["use_cross_validation"] = True

        # Act
        result = data_processing(sample_data, parameters)

        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_data_processing_split_mode(self, sample_data, parameters):
        """Testa data_processing no modo train/test split"""
        # Arrange
        parameters["use_cross_validation"] = False

        # Act
        result = data_processing(sample_data, parameters)
        n_result = 2

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == n_result
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_split_data_missing_columns(self, sample_data, parameters):
        """Testa comportamento com colunas ausentes"""
        # Arrange
        cleaned_data = load_clean_data(sample_data, parameters)
        parameters["target_column"] = "non_existent_column"

        # Act & Assert
        with pytest.raises(ValueError, match="Colunas necessárias não encontradas"):
            _split_data(cleaned_data, parameters)

    def test_prepare_cv_data_missing_columns(self, sample_data, parameters):
        """Testa comportamento com colunas ausentes no modo CV"""
        # Arrange
        cleaned_data = load_clean_data(sample_data, parameters)
        parameters["group_column"] = "non_existent_column"

        # Act & Assert
        with pytest.raises(ValueError, match="Colunas necessárias não encontradas"):
            _prepare_cv_data(cleaned_data, parameters)

    def test_empty_dataframe(self, parameters):
        """Testa comportamento com DataFrame vazio"""
        # Arrange
        empty_df = pd.DataFrame()

        # Act & Assert
        with pytest.raises(Exception):
            load_clean_data(empty_df, parameters)

    def test_features_dropped_correctly(self, sample_data, parameters):
        """Testa se as features especificadas são removidas"""
        # Arrange
        parameters["features_to_drop"] = ["vibration_x", "pressure"]
        cleaned_data = load_clean_data(sample_data, parameters)

        # Act
        cv_data = _prepare_cv_data(cleaned_data, parameters)

        # Assert
        assert "vibration_x" not in cv_data.columns
        assert "pressure" not in cv_data.columns
        assert "vibration_y" in cv_data.columns  # Não deve ser removida

    def test_data_integrity_after_processing(self, sample_data, parameters):
        """Testa a integridade dos dados após todo o processamento"""
        # Act
        cleaned_data = load_clean_data(sample_data, parameters)
        train_data, test_data = _split_data(cleaned_data, parameters)

        # Assert - verifica que não há dados corrompidos
        assert train_data.select_dtypes(include=[np.number]).notnull().all().all()
        assert test_data.select_dtypes(include=[np.number]).notnull().all().all()

        # Verifica que as colunas necessárias estão presentes
        required_cols = [parameters["target_column"], parameters["group_column"]]
        for col in required_cols:
            assert col in train_data.columns
            assert col in test_data.columns
