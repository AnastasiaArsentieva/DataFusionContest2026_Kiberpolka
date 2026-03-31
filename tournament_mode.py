import polars as pl
import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def evaluate_feature_set(file_path, X_all, y_all, skf):
    """Ускоренная проверка на CPU (берем только 20% данных для турнира)"""
    with open(file_path, "r") as f:
        features = [line.strip() for line in f.readlines()]

    print(f"\n>>> Тест набора: {file_path} ({len(features)} признаков)")

    # Чтобы не ждать вечность на CPU, берем подвыборку для оценки
    idx_sample = y_all.sample(frac=0.2, random_state=42).index
    X_subset = X_all.loc[idx_sample, features]
    y_subset = y_all.loc[idx_sample]
    cat_features = [c for c in features if "cat_feature" in c]

    train_idx, val_idx = next(skf.split(X_subset, y_subset.iloc[:, 0]))
    X_tr, X_vl = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
    y_tr, y_vl = y_subset.iloc[train_idx], y_subset.iloc[val_idx]

    # Используем CPU, но включаем все ядра (thread_count=-1)
    test_model = CatBoostClassifier(
        iterations=150, learning_rate=0.1, depth=4,
        task_type="CPU", thread_count=-1,
        loss_function='MultiLogloss', verbose=50, random_seed=42
    )

    test_model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), cat_features=cat_features)
    auc = roc_auc_score(y_vl, test_model.predict_proba(X_vl), average='macro')
    return auc, features


if __name__ == "__main__":
    # --- 1. ФОРМИРУЕМ ЯДРО ---
    feature_files = ["selected_features_elite_005.txt",
                     "selected_features_optimal_001.txt",
                     "selected_features_broad_0001.txt",
                     "selected_features_500.txt"]

    available_files = [f for f in feature_files if os.path.exists(f)]
    feature_sets = [set(line.strip() for line in open(f)) for f in available_files]

    if feature_sets:
        core_features = set.intersection(*feature_sets)
        core_file = "selected_features_CORE.txt"
        with open(core_file, "w") as f:
            for feat in sorted(core_features): f.write(f"{feat}\n")
        available_files = [core_file] + available_files

    # --- 2. ЗАГРУЗКА ---
    train_main = pl.read_parquet('data/train_main_features.parquet')
    train_extra = pl.read_parquet('data/train_extra_features.parquet')
    target = pl.read_parquet('data/train_target.parquet')
    full_train = train_main.join(train_extra, on="customer_id", how="left")
    y = target.drop("customer_id").to_pandas()
    target_cols = [c.replace("target_", "predict_") for c in y.columns]

    cat_cols = [c for c in full_train.columns if "cat_feature" in c]
    full_train = full_train.with_columns(pl.col(cat_cols).cast(pl.Int32).fill_null(-1))

    X_full_eval = full_train.drop("customer_id").to_pandas()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 3. ТУРНИР (CPU) ---
    best_auc, best_features, best_file = 0, None, ""
    for f_file in available_files:
        curr_auc, curr_feat = evaluate_feature_set(f_file, X_full_eval, y, skf)
        print(f"Результат {f_file}: AUC = {curr_auc:.6f}")
        if curr_auc > best_auc:
            best_auc, best_features, best_file = curr_auc, curr_feat, f_file

    print(f"\nПОБЕДИТЕЛЬ: {best_file} (AUC: {best_auc:.6f})")
    del X_full_eval

    # --- 4. ФИНАЛЬНОЕ ОБУЧЕНИЕ (CPU) ---
    X = full_train.select(best_features).to_pandas()
    test_main = pl.read_parquet('data/test_main_features.parquet')
    X_test = test_main.join(pl.read_parquet('data/test_extra_features.parquet'), on="customer_id", how="left") \
        .select(best_features).with_columns(
        pl.col([c for c in best_features if "cat_feature" in c]).cast(pl.Int32).fill_null(-1)).to_pandas()

    test_preds_total = np.zeros((len(X_test), y.shape[1]))
    cat_features = [c for c in best_features if "cat_feature" in c]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.iloc[:, 0])):
        print(f"\n=== ФИНАЛЬНЫЙ ФОЛД {fold + 1} (CPU) ===")
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.08, depth=6,
            task_type="CPU", thread_count=-1,
            loss_function='MultiLogloss', early_stopping_rounds=50,
            verbose=100, random_seed=42,
            save_snapshot=True, snapshot_file=f"snap_final_f{fold}.bkp"
        )
        model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), cat_features=cat_features)

        # Сохранение после каждого фолда
        model.save_model(f"model_fold_{fold}.cbm")
        test_preds_total += model.predict(X_test, prediction_type="RawFormulaVal")

        # Промежуточный сабмит
        sub = pd.DataFrame(test_preds_total / (fold + 1), columns=target_cols)
        sub.insert(0, 'customer_id', test_main['customer_id'].to_numpy())
        sub.to_parquet(f"sub_fold_{fold + 1}.parquet", index=False)

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО. Проверьте файлы .parquet")

