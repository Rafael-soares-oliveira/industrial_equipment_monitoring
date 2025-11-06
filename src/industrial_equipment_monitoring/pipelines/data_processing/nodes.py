import pandas as pd
from sklearn.model_selection import train_test_split

from industrial_equipment_monitoring.utils.logging_config import setup_logger

logger = setup_logger("DataProcessing")


def load_clean_data(
    raw_data: pd.DataFrame, parameters: dict
) -> tuple[dict[str, str | int | list], pd.DataFrame]:
    """Carrega e limpa os dados brutos removendo duplicatas e valores nulos.

    Args:
        raw_data (pd.DataFrame): DataFrame com dados brutos
        parameters (dict): Parâmetros de configuração

    Returns:
        pd.DataFrame: DataFrame limpo e processado
    """
    logger.info("Iniciando carregamento e limpeza dos dados")
    try:
        df = raw_data.copy()

        if df.empty:
            logger.error("DataFrame está vazio.")
            raise ValueError("DataFrame está vazio")

        logger.info(f"Dados brutos - Shape: {df.shape}")
        logger.info(f"Colunas disponíveis: {list(df.columns)}")

        # Remove duplicatas
        initial_count, cols = df.shape
        df = df.drop_duplicates()
        final_count = len(df)
        duplicates_removed = initial_count - final_count
        logger.info(f"Duplicatas removidas: {duplicates_removed}")

        # Remove valores nulos
        null_count_before = df.isnull().sum().sum()
        df = df.dropna()
        null_count_after = df.isnull().sum().sum()
        nulls_removed = null_count_before - null_count_after
        logger.info(f"Valores nulos removidos: {nulls_removed}")

        # Log de estatísticas pós-limpeza
        target_column = parameters["featured_data"]["target_column"]
        if target_column in df.columns:
            target_distribution = df[target_column].value_counts()
            logger.info(f"Distribuição da target após limpeza:\n{target_distribution}")
            logger.info(
                f"Proporção de classe positiva: {target_distribution.get(1, 0) / len(df):.3f}"
            )

        logger.info(f"Shape final após limpeza: {df.shape}")

        train = int((1 - parameters["split"]["test_size"]) * 100)
        test = int(parameters["split"]["test_size"] * 100)

        df_info = {
            "Colunas": cols,
            "Linhas": initial_count,
            "Valores duplicados": duplicates_removed,
            "Valores nulos": int(nulls_removed),
            "Colunas removidas": parameters["featured_data"]["features_to_drop"],
            "Coluna alvo": target_column,
            "Divisão treino/teste": f"{train}/{test}",
        }

        return df_info, df
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e


def split_data(
    cleaned_data: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide os dados em treino e teste mantendo grupos.

    Args:
        cleaned_data (pd.DataFrame): Dados limpos
        parameters (dict): Parâmetros de configuração

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dados de treino e teste
    """
    logger.info("Iniciando divisão dos dados em treino e teste")

    try:
        target_column = parameters["featured_data"]["target_column"]

        # Remove colunas não usadas
        features_to_drop = parameters["featured_data"]["features_to_drop"]
        df_clean = cleaned_data.drop(columns=features_to_drop, errors="ignore")
        logger.info(f"Colunas removidas: {features_to_drop}")

        # Validação de colunas necessárias
        required_columns = [target_column]
        missing_columns = [
            col for col in required_columns if col not in df_clean.columns
        ]
        if missing_columns:
            raise ValueError(f"Colunas necessárias não encontradas: {missing_columns}")

        # Separa features e target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        logger.info(f"Features shape: {X.shape}")

        # Split estratificado por grupo
        test_size = parameters["split"]["test_size"]
        random_state = parameters["split"]["random_state"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Prepara dados de treino
        train_data = X_train.copy()
        train_data[target_column] = y_train.values

        # Prepara dados de teste
        test_data = X_test.copy()
        test_data[target_column] = y_test.values

        logger.info(f"Treino: {train_data.shape}")
        logger.info(f"Teste: {test_data.shape}")
        logger.info(f"Defeitos no treino: {y_train.sum()} ({y_train.mean():.2%})")
        logger.info(f"Defeitos no teste: {y_test.sum()} ({y_test.mean():.2%})")

        return train_data, test_data

    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e
