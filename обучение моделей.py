import os, gc, pickle, optuna, time
import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier


class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, 41),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Пробрасываем данные через цепочку слоев
        return self.net(x)


BASE_CHECKPOINT_DIR = 'checkpoints_multioutput_500.v2'
os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)


def run_final_training(feature_path):
    if not os.path.exists(feature_path): return print(f"Файл {feature_path} не найден!")

    with open(feature_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]

    # 1. ЗАГРУЗКА
    print("Загрузка данных...")
    target = pl.read_parquet('data/train_target.parquet')
    y = target.drop("customer_id").to_pandas().astype(np.float32)
    target_cols = [c.replace("target_", "predict_") for c in y.columns]
    cat_features = [c for c in features if "cat_feature" in c]

    X = pl.read_parquet('data/train_main_features.parquet').join(
        pl.read_parquet('data/train_extra_features.parquet'), on="customer_id"
    ).select(features).with_columns([
        pl.col(cat_features).cast(pl.Int32).fill_null(-1),
        pl.exclude(cat_features).cast(pl.Float32)
    ]).to_pandas()

    X_test = pl.read_parquet('data/test_main_features.parquet').join(
        pl.read_parquet('data/test_extra_features.parquet'),
        on="customer_id").select(features).with_columns([
        pl.col(cat_features).cast(pl.Int32).fill_null(-1),
        pl.exclude(cat_features).cast(pl.Float32)]).to_pandas()

    test_ids = pl.read_parquet('data/test_main_features.parquet')['customer_id'].to_numpy()
    del target
    gc.collect()

    # 2. РЕСТАРТ
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_total = np.zeros((len(X_test), len(target_cols)))
    start_fold = 0
    for fold in range(5):
        if os.path.exists(os.path.join(BASE_CHECKPOINT_DIR, f'preds_fold_{fold}.pkl')):
            start_fold = fold + 1

    if start_fold > 0:
        with open(os.path.join(BASE_CHECKPOINT_DIR, f'preds_fold_{start_fold - 1}.pkl'), 'rb') as f:
            test_preds_total = pickle.load(f)
        print(f"РЕСТАРТ: Начинаем с фолда {start_fold + 1}")

    # 3. ГЛАВНЫЙ ЦИКЛ
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        if fold < start_fold: continue
        print(f"\n>>> ФОЛД {fold + 1} / 5 <<<")
        X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        # --- 1. CatBoost (GPU) ---
        cb_temp_path = os.path.join(BASE_CHECKPOINT_DIR, f'cb_temp_f{fold}.pkl')

        if os.path.exists(cb_temp_path):
            print(f"✅ Загрузка CatBoost из временного файла (Фолд {fold + 1})...")
            with open(cb_temp_path, 'rb') as f:
                p_cb_vl, p_cb_test = pickle.load(f)
        else:
            print(f"🐈 Обучение CatBoost (Фолд {fold + 1})...")
            cb = CatBoostClassifier(
                iterations=8000,
                learning_rate=0.01,
                depth=7,
                #l2_leaf_reg=30,  # Сильная защита от шума
                #random_strength=2,  # Добавляем хаоса, чтобы не зацикливалась
                #bagging_temperature=0.2,  # Более агрессивный сэмплинг
                #max_ctr_complexity=3,  # Усложняем работу с категориями

                task_type="GPU",
                devices='0',
                gpu_ram_part=0.7,
                leaf_estimation_iterations=1,
                max_bin=32,
                model_size_reg=0,
                bootstrap_type='Bernoulli',
                subsample=0.5,
                loss_function='MultiLogloss',
                eval_metric='MultiLogloss',
                early_stopping_rounds=50,
                random_seed=42,
                verbose=100,
                save_snapshot=True,
                train_dir=BASE_CHECKPOINT_DIR,
                snapshot_file=f"cb_snapshot_fold_{fold}.bkp",
                snapshot_interval=60
            )

            cb.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), cat_features=cat_features)

            # Предсказания
            p_cb_vl = cb.predict_proba(X_vl)
            p_cb_test = cb.predict_proba(X_test)

            # СОХРАНЕНИЕ В .PKL (теперь аудит его увидит)
            print(f"💾 Сохранение CatBoost в {cb_temp_path}...")
            with open(cb_temp_path, 'wb') as f:
                pickle.dump((p_cb_vl, p_cb_test), f)

            del cb
            gc.collect()

            # Локальный импорт прямо перед вызовом (защита от вылета)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Очищаем VRAM для следующих моделей

        # --- 2. LightGBM (CPU - МИКСОГЛУБИНА + СЕЙВЫ) ---
        lgb_temp_path = os.path.join(BASE_CHECKPOINT_DIR, f'lgb_temp_f{fold}.pkl')
        lgb_partial_path = os.path.join(BASE_CHECKPOINT_DIR, f'lgb_partial_f{fold}.npz')

        depth_5_targets = ['predict_1_5', 'predict_2_1', 'predict_2_4', 'predict_2_5', 'predict_2_7', 'predict_2_8',
                           'predict_3_5', 'predict_5_1', 'predict_5_2', 'predict_6_5', 'predict_7_3', 'predict_8_3',
                           'predict_9_1', 'predict_9_5']

        if os.path.exists(lgb_temp_path):
            print(f"✅ Загрузка LGBM из финального файла (Фолд {fold + 1})...")
            with open(lgb_temp_path, 'rb') as f:
                p_lgb_vl, p_lgb_test = pickle.load(f)
        else:
            # 1. ПРОВЕРКА ЧАСТИЧНОГО СОХРАНЕНИЯ
            if os.path.exists(lgb_partial_path):
                print(f"🔄 Найдено частичное сохранение LGBM (Фолд {fold + 1}). Подгружаю прогресс...")
                data = np.load(lgb_partial_path)
                p_lgb_vl, p_lgb_test = data['vl'], data['ts']
                start_col_idx = int(data['last_idx']) + 1
                print(f"▶️ Продолжаем с колонки {start_col_idx + 1}/41")
            else:
                print(f"🌲 Обучение LightGBM (MixDepth) - Всего {len(target_cols)} колонок")
                p_lgb_vl = np.zeros((len(X_vl), len(target_cols)))
                p_lgb_test = np.zeros((len(X_test), len(target_cols)))
                start_col_idx = 0

            # 2. ЦИКЛ ПО СТОЛБЦАМ
            for i, col in enumerate(target_cols):
                if i < start_col_idx: continue  # Пропускаем уже готовые

                start_time = time.time()
                curr_depth, curr_leaves, tag = (5, 24, "D5") if col in depth_5_targets else (8, 128, "D8")

                print(f"  [{i + 1}/41] LGBM ({tag}): {col} ... ", end="", flush=True)

                lgb = LGBMClassifier(
                    metric="auc", n_estimators=10000, learning_rate=0.02,
                    max_depth=curr_depth, num_leaves=curr_leaves,
                    device="cpu", n_jobs=4, min_child_samples=20,
                    random_state=42, verbose=-1
                )

                lgb.fit(X_tr, y_tr.iloc[:, i],
                        eval_set=[(X_vl, y_vl.iloc[:, i])],
                        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)])

                p_lgb_vl[:, i] = lgb.predict_proba(X_vl)[:, 1]
                p_lgb_test[:, i] = lgb.predict_proba(X_test)[:, 1]

                # СОХРАНЕНИЕ ПОСЛЕ КАЖДОЙ КОЛОНКИ
                np.savez(lgb_partial_path, vl=p_lgb_vl, ts=p_lgb_test, last_idx=i)

                print(f"Done! ({time.time() - start_time:.1f}s)")
                del lgb
                gc.collect()

            # 3. ФИНАЛЬНОЕ СОХРАНЕНИЕ В .PKL
            with open(lgb_temp_path, 'wb') as f:
                pickle.dump((p_lgb_vl, p_lgb_test), f)

            # Удаляем временный файл прогресса
            if os.path.exists(lgb_partial_path):
                os.remove(lgb_partial_path)

        # --- 3. XGBoost (GPU - ЧИСТЫЙ И БЫСТРЫЙ + СЕЙВЫ) ---
        xgb_temp_path = os.path.join(BASE_CHECKPOINT_DIR, f'xgb_temp_f{fold}.pkl')
        xgb_partial_path = os.path.join(BASE_CHECKPOINT_DIR, f'xgb_partial_f{fold}.npz')

        if os.path.exists(xgb_temp_path):
            print(f"✅ Загрузка XGBoost из финального файла (Фолд {fold + 1})...")
            with open(xgb_temp_path, 'rb') as f:
                p_xgb_vl, p_xgb_test = pickle.load(f)
        else:
            # 1. ПРОВЕРКА ЧАСТИЧНОГО СОХРАНЕНИЯ
            if os.path.exists(xgb_partial_path):
                print(f"🔄 Найдено частичное сохранение XGBoost (Фолд {fold + 1}). Подгружаю...")
                data = np.load(xgb_partial_path)
                p_xgb_vl, p_xgb_test = data['vl'], data['ts']
                start_col_idx = int(data['last_idx']) + 1
                print(f"▶️ Продолжаем с колонки {start_col_idx + 1}/41")
            else:
                print(f"⚡ Обучение XGBoost (GPU) - Всего {len(target_cols)} колонок")
                p_xgb_vl = np.zeros((len(X_vl), len(target_cols)))
                p_xgb_test = np.zeros((len(X_test), len(target_cols)))
                start_col_idx = 0

            # 2. ЦИКЛ ПО СТОЛБЦАМ
            for i, col in enumerate(target_cols):
                if i < start_col_idx: continue

                start_time = time.time()
                print(f"  [{i + 1}/41] XGB: {col} ... ", end="", flush=True)

                xgb = XGBClassifier(
                    n_estimators=10000,
                    learning_rate=0.02,
                    max_depth=7,
                    tree_method='hist',
                    device='cuda',
                    eval_metric='auc',
                    early_stopping_rounds=100,
                    subsample=0.8,
                    colsample_bytree=0.1,
                    random_state=42,
                    verbosity=0
                )

                xgb.fit(X_tr, y_tr.iloc[:, i], eval_set=[(X_vl, y_vl.iloc[:, i])], verbose=False)

                p_xgb_vl[:, i] = xgb.predict_proba(X_vl)[:, 1]
                p_xgb_test[:, i] = xgb.predict_proba(X_test)[:, 1]

                # СОХРАНЕНИЕ ПОСЛЕ КАЖДОЙ КОЛОНКИ
                np.savez(xgb_partial_path, vl=p_xgb_vl, ts=p_xgb_test, last_idx=i)

                print(f"Done! ({time.time() - start_time:.1f}s)")
                del xgb
                gc.collect()
                # Очистка видеопамяти после каждой колонки
                import torch
                torch.cuda.empty_cache()

            # 3. ФИНАЛЬНОЕ СОХРАНЕНИЕ В .PKL
            with open(xgb_temp_path, 'wb') as f:
                pickle.dump((p_xgb_vl, p_xgb_test), f)

            # Удаляем временный файл прогресса
            if os.path.exists(xgb_partial_path):
                os.remove(xgb_partial_path)

        # --- 4. Neural Network (PyTorch MLP) ---
        # --- 4. Neural Network (PyTorch MLP) ---
        import torch
        nn_temp_path = os.path.join(BASE_CHECKPOINT_DIR, f'nn_temp_f{fold}.pkl')
        if os.path.exists(nn_temp_path):
            print("✅ Загрузка Нейросети из временного файла...")
            with open(nn_temp_path, 'rb') as f:
                p_nn_vl, p_nn_test = pickle.load(f)
        else:
            print("🧠 Обучение Нейросети (MLP)...")
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import QuantileTransformer

            # 1. Функция подготовки данных (теперь внутри цикла)
            def prepare_nn_data(df_tr, df_vl, df_ts):
                # Копируем, чтобы не испортить исходные DF для бустингов
                tr, vl, ts = df_tr.copy(), df_vl.copy(), df_ts.copy()

                # Кодируем категории в числа (LabelEncoding)
                for col in tr.select_dtypes(['category', 'object']).columns:
                    # Создаем маппинг на основе трейна
                    codes, uniques = pd.factorize(tr[col])
                    tr[col] = codes
                    # Применяем тот же маппинг к валидации и тесту
                    vl[col] = pd.Categorical(vl[col], categories=uniques).codes
                    ts[col] = pd.Categorical(ts[col], categories=uniques).codes

                # Самое важное: НОРМАЛИЗАЦИЯ
                print("🔄 Применяю QuantileTransformer...")
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)

                # Обучаем на Трейне (fit_transform), применяем к остальным (transform)
                X_tr_sc = scaler.fit_transform(tr.fillna(0))
                X_vl_sc = scaler.transform(vl.fillna(0))
                X_ts_sc = scaler.transform(ts.fillna(0))

                return X_tr_sc.astype(np.float32), X_vl_sc.astype(np.float32), X_ts_sc.astype(np.float32)

            # ВЫЗЫВАЕМ ОДИН РАЗ ДЛЯ ВСЕХ ТРЕХ
            X_tr_nn, X_vl_nn, X_ts_nn = prepare_nn_data(X_tr, X_vl, X_test)

            # 2. Инициализация модели (TabularNN теперь с 1024 нейронами)
            model = DeepMLP(X_tr_nn.shape[1]).cuda()
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            # Добавляем плавное снижение скорости обучения
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

            #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) - было
            #criterion = nn.BCELoss() - было

            # Используем DataLoader
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_tr_nn), torch.FloatTensor(y_tr.values)),
                batch_size=2048, shuffle=True
            )

            # Обучение (30 эпох достаточно для начала)
            for epoch in range(30):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    outputs = model(xb.cuda())
                    loss = criterion(outputs, yb.cuda())
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch + 1}/30] Loss: {loss.item():.4f}")

                scheduler.step()

            model.eval()
            with torch.no_grad():
                p_nn_vl = model(torch.FloatTensor(X_vl_nn).cuda()).cpu().numpy()
                p_nn_test = model(torch.FloatTensor(X_ts_nn).cuda()).cpu().numpy()

            with open(nn_temp_path, 'wb') as f:
                pickle.dump((p_nn_vl, p_nn_test), f)

            # ОЧИСТКА ПАМЯТИ
            del model, X_tr_nn, X_vl_nn, X_ts_nn
            gc.collect()
            torch.cuda.empty_cache()


        # --- 5. Random Forest (🌲 Стабилизатор с поколоночным сохранением) ---
        rf_temp_path = os.path.join(BASE_CHECKPOINT_DIR, f'rf_temp_f{fold}.pkl')
        rf_partial_path = os.path.join(BASE_CHECKPOINT_DIR, f'rf_partial_f{fold}.npz')

        if os.path.exists(rf_temp_path):
            print(f"✅ Загрузка Random Forest из финального файла (Фолд {fold + 1})...")
            with open(rf_temp_path, 'rb') as f:
                p_rf_vl, p_rf_test = pickle.load(f)
        else:
            # 1. Загружаем прогресс или создаем нули
            if os.path.exists(rf_partial_path):
                data = np.load(rf_partial_path)
                p_rf_vl, p_rf_test = data['vl'], data['ts']
                start_col = int(data['last_idx']) + 1
                print(f"🔄 Найдено частичное сохранение. Продолжаем с колонки {start_col + 1}/41")
            else:
                print(f"🌲 Обучение Random Forest (Фолд {fold + 1})...")
                p_rf_vl = np.zeros((len(X_vl), 41))
                p_rf_test = np.zeros((len(X_test), 41))
                start_col = 0

            # 2. Твоя рабочая подготовка данных
            X_tr_rf = X_tr.fillna(0).reset_index(drop=True).select_dtypes(exclude=['category', 'object'])
            X_vl_rf = X_vl.fillna(0).reset_index(drop=True).select_dtypes(exclude=['category', 'object'])
            X_test_rf = X_test.fillna(0).reset_index(drop=True).select_dtypes(exclude=['category', 'object'])
            y_tr_rf = y_tr.reset_index(drop=True)

            # 3. Цикл обучения
            for i in range(start_col, 41):
                start_time_col = time.time()
                rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
                rf.fit(X_tr_rf, y_tr_rf.iloc[:, i])

                # Защита [:, -1] берет вероятность "1", даже если в данных 3 класса
                p_rf_vl[:, i] = rf.predict_proba(X_vl_rf)[:, -1]
                p_rf_test[:, i] = rf.predict_proba(X_test_rf)[:, -1]

                # СОХРАНЕНИЕ ПОСЛЕ КАЖДОЙ КОЛОНКИ
                np.savez(rf_partial_path, vl=p_rf_vl, ts=p_rf_test, last_idx=i)

                print(f"  [{i + 1}/41] RF: Done! ({time.time() - start_time_col:.1f}s)")

                if (i + 1) % 10 == 0: gc.collect()

            # 4. Финальный сейв в .pkl для Optuna
            with open(rf_temp_path, 'wb') as f:
                pickle.dump((p_rf_vl, p_rf_test), f)

            # Удаляем временный файл прогресса
            if os.path.exists(rf_partial_path):
                os.remove(rf_partial_path)

            del X_tr_rf, X_vl_rf, X_test_rf, y_tr_rf
            gc.collect()

        # --- 6. OPTUNA (ФИНАЛЬНЫЙ СУПЕР-АНСАМБЛЬ) ---
        print("\n🎯 Глобальная оптимизация весов (5 моделей)...")
        import torch
        # Считаем AUC для каждой модели отдельно для статистики
        auc_cb = roc_auc_score(y_vl, p_cb_vl, average="macro")
        auc_xgb = roc_auc_score(y_vl, p_xgb_vl, average="macro")
        auc_lgb = roc_auc_score(y_vl, p_lgb_vl, average="macro")
        auc_nn = roc_auc_score(y_vl, p_nn_vl, average="macro")
        auc_rf = roc_auc_score(y_vl, p_rf_vl, average="macro")

        print(f"📊 AUC фолда: CB={auc_cb:.5f}, XGB={auc_xgb:.5f}, LGB={auc_lgb:.5f}, NN={auc_nn:.5f}, RF={auc_rf:.5f}")

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: roc_auc_score(y_vl, (
                t.suggest_float("c", 0.1, 1.0) * p_cb_vl +
                t.suggest_float("x", 0.1, 1.0) * p_xgb_vl +
                t.suggest_float("l", 0.1, 1.0) * p_lgb_vl +
                t.suggest_float("n", 0.05, 0.5) * p_nn_vl +
                t.suggest_float("r", 0.05, 0.5) * p_rf_vl
        ), average="macro"), n_trials=100)

        # Извлекаем веса и нормализуем их
        w = study.best_params
        sw = sum(w.values())
        norm_w = {k: round(v / sw, 4) for k, v in w.items()}
        best_auc = study.best_value

        print(
            f"✅ ВЕСА: CB={norm_w['c']:.2f}, XGB={norm_w['x']:.2f}, LGB={norm_w['l']:.2f}, NN={norm_w['n']:.2f}, RF={norm_w['r']:.2f}")
        print(f"🏆 Best Ensemble AUC: {best_auc:.5f}")

        # --- СОХРАНЕНИЕ СТАТИСТИКИ В CSV ---
        log_file = "ensemble_history.csv"
        log_data = {
            'Fold': fold + 1,
            'Best_AUC': round(best_auc, 6),
            'W_CB': norm_w['c'], 'W_XGB': norm_w['x'], 'W_LGB': norm_w['l'], 'W_NN': norm_w['n'], 'W_RF': norm_w['r'],
            'AUC_CB': round(auc_cb, 5), 'AUC_XGB': round(auc_xgb, 5), 'AUC_LGB': round(auc_lgb, 5),
            'AUC_NN': round(auc_nn, 5), 'AUC_RF': round(auc_rf, 5)
        }

        log_df = pd.DataFrame([log_data])
        if not os.path.exists(log_file):
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode='a', header=False, index=False)

        # Смешиваем всё в итоговый результат фолда
        fold_blend = (w['c'] * p_cb_test + w['x'] * p_xgb_test + w['l'] * p_lgb_test + w['n'] * p_nn_test + w[
            'r'] * p_rf_test) / sw
        test_preds_total += fold_blend / 5

        # Сохраняем прогресс фолда
        with open(os.path.join(BASE_CHECKPOINT_DIR, f'preds_fold_{fold}.pkl'), 'wb') as f:
            pickle.dump(test_preds_total, f)

        del X_tr, X_vl, y_tr, y_vl
        gc.collect()

    # 4. ФИНАЛ
    pd.DataFrame(test_preds_total, columns=target_cols).assign(customer_id=test_ids).to_parquet(
        "submit_5folds_500_final.parquet", index=False)
    print("\nГОТОВО!")


if __name__ == "__main__":
    run_final_training("selected_features_500.txt")
