import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

if __name__ == "__main__":
    print("--- ШАГ 1: ГЛУБОКИЙ ОТБОР ПРИЗНАКОВ (2000 ITERATIONS MODE) ---")

    # 1. Загрузка данных (Polars -> Pandas)
    train_main = pl.read_parquet('data/train_main_features.parquet')
    train_extra = pl.read_parquet('data/train_extra_features.parquet')
    target = pl.read_parquet('data/train_target.parquet')

    X = train_main.join(train_extra, on="customer_id", how="left").drop("customer_id").to_pandas()
    y = target.drop("customer_id").to_pandas()

    print(f"Анализ {X.shape[1]} признаков на 50% данных...")
    X_sample = X.sample(frac=0.5, random_state=42)
    y_sample = y.loc[X_sample.index]

    # 2. Настройка модели на 2000 итераций
    fs_model = CatBoostClassifier(
        iterations=2000,  
        learning_rate=0.05,  
        depth=6,
        rsm=0.1,
        loss_function='MultiLogloss',
        early_stopping_rounds=100,  
        verbose=100,
        thread_count=-1,
        random_seed=42
    )

    # Обучаем один раз до победного
    fs_model.fit(X_sample, y_sample)

    # 3. ПОЛУЧЕНИЕ НАКОПИТЕЛЬНОЙ ВАЖНОСТИ
    fi = pd.Series(fs_model.feature_importances_, index=X.columns).sort_values(ascending=False)


    # Функция для сохранения списков по разным критериям "силы" признака
    def save_list(name, features):
        with open(f"selected_features_{name}.txt", "w") as f:
            for feat in features:
                f.write(f"{feat}\n")
        print(f"Сохранено {name}: {len(features)} признаков")


    # Формируем 3 уровня качества (Накопительная важность)
    # Порог 0.05 - Самое мощное "Ядро" (Elite Core)
    save_list("elite_005", fi[fi > 0.05].index.tolist())

    # Порог 0.01 - Оптимальный баланс для 0.95 (Recommended)
    save_list("optimal_001", fi[fi > 0.01].index.tolist())

    # Порог 0.001 - Широкий охват (для сложных ансамблей)
    save_list("broad_0001", fi[fi > 0.001].index.tolist())

    print("\nГлубокий анализ завершен. Все три списка созданы.")
