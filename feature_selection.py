import polars as pl
import pandas as pd
from catboost import CatBoostClassifier

if __name__ == "__main__":
    print("--- ШАГ 1: ГЛУБОКИЙ ОТБОР ПРИЗНАКОВ ---")

    # Используем Polars для склейки 2446 признаков
    train_main = pl.read_parquet('data/train_main_features.parquet')
    train_extra = pl.read_parquet('data/train_extra_features.parquet')
    target = pl.read_parquet('data/train_target.parquet')

    # Объединяем всё и переводим в Pandas только в конце
    X = train_main.join(train_extra, on="customer_id", how="left").drop("customer_id").to_pandas()
    y = target.drop("customer_id").to_pandas()

    print(f"Анализ {X.shape[1]} признаков на 50% данных...")
    X_sample = X.sample(frac=0.5, random_state=42)
    y_sample = y.loc[X_sample.index]

    # Обозначаем парметры модели
    fs_model = CatBoostClassifier(
        iterations=500,  
        learning_rate=0.1,
        depth=6,  
        rsm=0.1,  
        loss_function='MultiLogloss',
        verbose=100,
        thread_count=-1,
        random_seed=42
    )

    fs_model.fit(X_sample, y_sample)

    # Берем все признаки, которые дали хотя бы какой-то профит (важность > 0)
    fi = pd.Series(fs_model.feature_importances_, index=X.columns)
    selected_features = fi[fi > 0.001].sort_values(ascending=False).index.tolist()

    with open("selected_features.txt", "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    print(f"Готово! Из 2446 отобрано {len(selected_features)} признаков.")
    
