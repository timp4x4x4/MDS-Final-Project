import os, time, json, gc, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# XGBoost
import xgboost as xgb

warnings.filterwarnings('ignore')

# GPU 檢查
print("="*60)
print("環境檢查")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"XGBoost version: {xgb.__version__}")
print("="*60)


def reduce_memory_usage(df, verbose=True):
    """通過改變數據類型來減少DataFrame的記憶體使用"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'記憶體使用減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%')
        print(f'{start_mem:.2f} MB --> {end_mem:.2f} MB')
    
    return df


def clean_memory():
    """清理記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_all_data(file_path, sample_frac=1.0):
    """載入全美國的資料"""
    print(f"\n載入資料: {file_path}")
    
    # 定義需要的欄位（排除會看答案的欄位）
    important_cols = [
        'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng',
        'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
        'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset', 'State',
        'City', 'County'
    ]
    
    # 載入資料
    print(f"載入 {sample_frac*100:.0f}% 的全美資料...")
    df = pd.read_csv(file_path, usecols=lambda x: x in important_cols)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"全美資料大小: {df.shape}")
    print(f"記憶體使用: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # 轉換日期時間
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    
    # 顯示州分布（前10個）
    print("\n州分布 (前10):")
    state_counts = df['State'].value_counts().head(10)
    for state, count in state_counts.items():
        print(f"  {state}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # 顯示目標變數分布
    print("\n目標變數分布:")
    severity_counts = df['Severity'].value_counts().sort_index()
    for sev, count in severity_counts.items():
        print(f"  Severity {sev}: {count:,} ({count/len(df)*100:.2f}%)")
    
    return df


def split_data_by_time(df):
    """按時間分割資料：2016-2022年3月 vs 2022年4月-2023年3月"""
    # 設定時間分割點
    train_end_date = pd.Timestamp('2022-03-31 23:59:59')
    test_start_date = pd.Timestamp('2022-04-01 00:00:00')
    
    # 分割資料
    df_train = df[df['Start_Time'] <= train_end_date].copy()
    df_test = df[df['Start_Time'] >= test_start_date].copy()
    
    print(f"\n訓練資料 (2016-01 到 2022-03): {len(df_train):,} 筆")
    print(f"測試資料 (2022-04 到 2023-03): {len(df_test):,} 筆")
    
    # 顯示訓練資料的時間範圍
    print(f"\n訓練資料時間範圍:")
    print(f"  開始: {df_train['Start_Time'].min()}")
    print(f"  結束: {df_train['Start_Time'].max()}")
    
    # 顯示測試資料的時間範圍
    print(f"\n測試資料時間範圍:")
    print(f"  開始: {df_test['Start_Time'].min()}")
    print(f"  結束: {df_test['Start_Time'].max()}")
    
    # 顯示測試資料中各州的分布
    print(f"\n測試資料州分布 (前10):")
    test_state_counts = df_test['State'].value_counts().head(10)
    for state, count in test_state_counts.items():
        print(f"  {state}: {count:,} ({count/len(df_test)*100:.2f}%)")
    
    return df_train, df_test


def advanced_preprocessing(df):
    """進階前處理：包含所有特徵工程"""
    df_copy = df.copy()
    
    # 提取時間特徵
    df_copy['Hour'] = df_copy['Start_Time'].dt.hour
    df_copy['DayOfWeek'] = df_copy['Start_Time'].dt.dayofweek
    df_copy['Month'] = df_copy['Start_Time'].dt.month
    df_copy['DayOfMonth'] = df_copy['Start_Time'].dt.day
    df_copy['Year'] = df_copy['Start_Time'].dt.year
    
    # 額外的特徵工程
    # 1. 是否週末
    df_copy['IsWeekend'] = (df_copy['DayOfWeek'] >= 5).astype(int)
    
    # 2. 是否尖峰時段（全美通勤時間）
    df_copy['IsRushHour'] = df_copy['Hour'].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )
    
    # 3. 時段分類
    df_copy['TimeOfDay'] = pd.cut(df_copy['Hour'], 
                                  bins=[-1, 6, 12, 18, 24], 
                                  labels=[0, 1, 2, 3]).astype(int)
    
    # 4. 季節
    df_copy['Season'] = df_copy['Month'].apply(
        lambda x: 0 if x in [12, 1, 2] else  # 冬季
                  1 if x in [3, 4, 5] else   # 春季
                  2 if x in [6, 7, 8] else   # 夏季
                  3  # 秋季
    )
    
    # 5. 天氣分類
    if 'Weather_Condition' in df_copy.columns:
        def categorize_weather(condition):
            if pd.isna(condition):
                return 0
            condition = str(condition).lower()
            if any(word in condition for word in ['clear', 'fair']):
                return 1
            elif any(word in condition for word in ['cloud', 'overcast']):
                return 2
            elif any(word in condition for word in ['rain', 'drizzle']):
                return 3
            elif any(word in condition for word in ['snow', 'sleet']):
                return 4
            elif any(word in condition for word in ['fog', 'mist']):
                return 5
            elif any(word in condition for word in ['storm', 'thunder']):
                return 6
            else:
                return 7
        
        df_copy['Weather_Category'] = df_copy['Weather_Condition'].apply(categorize_weather)
        df_copy = df_copy.drop('Weather_Condition', axis=1)
    
    # 6. 處理缺失值
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Severity']:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # 7. 類別變數編碼
    label_encoders = {}
    categorical_cols = ['Sunrise_Sunset', 'City', 'County', 'State']
    for col in categorical_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('Unknown')
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le
    
    # 8. 布林型欄位轉換
    bool_cols = df_copy.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_copy[col] = df_copy[col].astype(int)
    
    # 刪除不需要的欄位
    df_copy = df_copy.drop(['Start_Time'], axis=1, errors='ignore')
    
    return df_copy, label_encoders


def apply_mixed_sampling(X_train, y_train):
    """應用混合採樣策略 - 強化版"""
    print("\n應用混合採樣策略...")
    
    # 計算各類別數量和比例
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_samples = len(y_train)
    
    print("原始分布:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  類別 {cls}: {cnt:,} ({cnt/total_samples*100:.2f}%)")
    
    # 計算不平衡比率
    imbalance_ratio = max(counts) / min(counts)
    print(f"不平衡比率: {imbalance_ratio:.2f}")
    
    # 動態調整目標數量
    # 如果極度不平衡（比率>10），使用更激進的策略
    if imbalance_ratio > 10:
        target_count = int(np.percentile(counts, 75))
    else:
        target_count = int(np.median(counts) * 1.5)
    
    print(f"目標樣本數: {target_count:,}")
    
    # 第一步：對多數類別欠採樣
    undersample_strategy = {}
    for cls, cnt in class_counts.items():
        if cnt > target_count * 1.5:  # 只對超過目標1.5倍的類別欠採樣
            undersample_strategy[cls] = int(target_count * 1.5)
        else:
            undersample_strategy[cls] = cnt
    
    if undersample_strategy:
        print("執行欠採樣...")
        rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
        X_temp, y_temp = rus.fit_resample(X_train, y_train)
    else:
        X_temp, y_temp = X_train, y_train
    
    # 中間步驟：顯示欠採樣後的分布
    temp_unique, temp_counts = np.unique(y_temp, return_counts=True)
    temp_class_counts = dict(zip(temp_unique, temp_counts))
    
    # 第二步：對少數類別過採樣
    oversample_strategy = {}
    min_samples = min(temp_counts)
    
    for cls, cnt in temp_class_counts.items():
        if cnt < target_count:
            # 確保少數類別至少達到目標數量
            oversample_strategy[cls] = target_count
        else:
            oversample_strategy[cls] = cnt
    
    if oversample_strategy and len(set(oversample_strategy.values())) > 1:
        print("執行過採樣...")
        ros = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_temp, y_temp)
    else:
        X_resampled, y_resampled = X_temp, y_temp
    
    # 顯示最終分布
    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    total_new = len(y_resampled)
    
    print("\n採樣後分布:")
    for cls, cnt in zip(unique_new, counts_new):
        print(f"  類別 {cls}: {cnt:,} ({cnt/total_new*100:.2f}%)")
    
    new_imbalance_ratio = max(counts_new) / min(counts_new)
    print(f"新的不平衡比率: {new_imbalance_ratio:.2f}")
    print(f"總樣本數變化: {total_samples:,} → {total_new:,}")
    
    return X_resampled, y_resampled


def train_xgboost_gpu(X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    """XGBoost GPU 訓練器（優化參數以減少訓練時間）"""
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        # GPU 設定
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'max_bin': 128,  # 降低以加快速度
        # 調整參數以加快訓練
        'max_depth': 6,  # 降低深度
        'learning_rate': 0.1,  # 提高學習率
        'n_estimators': 5000,  # 減少樹的數量
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'lambda': 1.0,
        'eval_metric': ['mlogloss', 'merror'],
        'early_stopping_rounds': 50,  # 提早停止
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_test, y_test)]
    if X_val is not None:
        eval_set.append((X_val, y_val))
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    train_time = time.time() - start
    
    preds = model.predict(X_test)
    return model, preds, train_time


def run_experiment(df_train, df_test):
    """執行全美資料實驗"""
    results = {}
    
    # 處理訓練資料
    print("\n處理訓練資料...")
    df_train_processed, label_encoders = advanced_preprocessing(df_train)
    
    # 確保只有 Severity 1-4
    df_train_processed = df_train_processed[df_train_processed['Severity'].isin([1,2,3,4])].dropna()
    
    # 處理 object 類型欄位
    obj_cols = df_train_processed.select_dtypes(include='object').columns
    df_train_processed[obj_cols] = df_train_processed[obj_cols].astype('category').apply(lambda s: s.cat.codes)
    
    X = df_train_processed.drop('Severity', axis=1).values
    y = df_train_processed['Severity'].values - 1  # 轉換為 0-3
    
    # 顯示原始類別分布
    print("\n訓練資料原始分布:")
    unique_orig, counts_orig = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique_orig, counts_orig):
        print(f"  Severity {cls+1}: {cnt:,} ({cnt/len(y)*100:.2f}%)")
    
    # 分割訓練、驗證、測試集
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp
    )
    
    # 應用混合採樣
    X_train_s, y_train_s = apply_mixed_sampling(X_train, y_train)
    
    # 訓練模型
    print("\n--- 訓練 XGBoost 模型 (全美資料) ---")
    model, predictions, train_time = train_xgboost_gpu(
        X_train_s, X_test, y_train_s, y_test,
        X_val=X_val, y_val=y_val
    )
    
    # 評估指標 - 詳細版本
    acc = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_macro = f1_score(y_test, predictions, average='macro')
    f1_micro = f1_score(y_test, predictions, average='micro')
    bal_acc = balanced_accuracy_score(y_test, predictions)
    
    # 計算每個類別的 F1 分數
    f1_per_class = f1_score(y_test, predictions, average=None)
    
    print(f"\n========== 模型表現評估 ==========")
    print(f"整體指標:")
    print(f"  準確率 (Accuracy): {acc:.4f}")
    print(f"  平衡準確率 (Balanced Accuracy): {bal_acc:.4f}")
    print(f"  加權 F1 分數 (Weighted F1): {f1_weighted:.4f}")
    print(f"  宏平均 F1 分數 (Macro F1): {f1_macro:.4f}")
    print(f"  微平均 F1 分數 (Micro F1): {f1_micro:.4f}")
    print(f"  訓練時間: {train_time:.1f} 秒")
    
    print(f"\n各類別 F1 分數:")
    for i, f1 in enumerate(f1_per_class):
        severity = i + 1
        print(f"  Severity {severity}: {f1:.4f}")
    
    # 詳細的分類報告
    print("\n詳細分類報告:")
    target_names = [f'Severity {i+1}' for i in range(4)]
    print(classification_report(y_test, predictions, target_names=target_names))
    
    # 混淆矩陣
    print("\n混淆矩陣:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # 儲存所有結果
    results['model'] = model
    results['label_encoders'] = label_encoders
    results['feature_columns'] = list(df_train_processed.drop('Severity', axis=1).columns)
    results['metrics'] = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_per_class': f1_per_class.tolist(),
        'train_time': train_time,
        'confusion_matrix': cm.tolist()
    }
    
    # 視覺化混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('混淆矩陣 - 全美事故嚴重度預測')
    plt.xlabel('預測類別')
    plt.ylabel('真實類別')
    plt.tight_layout()
    plt.savefig('confusion_matrix_us.png', dpi=300)
    plt.show()
    
    # 特徵重要性
    feature_importance = model.feature_importances_
    feature_names = list(df_train_processed.drop('Severity', axis=1).columns)
    
    # 顯示前 20 個重要特徵
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('重要性分數')
    plt.title('前 20 個重要特徵 (全美模型)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_us.png', dpi=300)
    plt.show()
    
    print("\n前 10 個最重要特徵:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return results, df_test


def create_kepler_predictions_ca(df_test, model, feature_columns, label_encoders):
    """使用全美模型預測，但只輸出 California 的結果"""
    print("\n準備 California 2022.04-2023.03 Kepler.gl 預測資料...")
    
    # 先篩選出 California 的資料
    df_test_ca = df_test[df_test['State'] == 'CA'].copy()
    print(f"California 測試資料數量: {len(df_test_ca):,}")
    
    # 處理測試資料
    df_test_processed, _ = advanced_preprocessing(df_test_ca)
    
    # 應用相同的 label encoders
    for col, le in label_encoders.items():
        if col in df_test_processed.columns:
            # 處理未見過的類別
            df_test_processed[col] = df_test_processed[col].fillna('Unknown')
            df_test_processed[col] = df_test_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # 處理 object 類型
    obj_cols = df_test_processed.select_dtypes(include='object').columns
    df_test_processed[obj_cols] = df_test_processed[obj_cols].astype('category').apply(lambda s: s.cat.codes)
    
    # 刪除 Severity（如果存在）
    if 'Severity' in df_test_processed.columns:
        df_test_processed = df_test_processed.drop('Severity', axis=1)
    
    # 確保所有訓練時的特徵都存在
    for col in feature_columns:
        if col not in df_test_processed.columns:
            df_test_processed[col] = 0
    
    # 按訓練時的順序排列欄位
    df_test_processed = df_test_processed[feature_columns]
    
    # 批次預測（避免記憶體問題）
    batch_size = 50000
    all_probs = []
    
    for i in tqdm(range(0, len(df_test_processed), batch_size), desc="批次預測"):
        batch = df_test_processed.iloc[i:i+batch_size]
        batch_probs = model.predict_proba(batch.values)
        all_probs.append(batch_probs)
    
    all_probs = np.vstack(all_probs)
    
    # 計算風險分數和等級
    risk_scores = all_probs.max(axis=1)
    predicted_severity = all_probs.argmax(axis=1) + 1  # 轉回 1-4
    risk_levels = np.where(risk_scores > 0.7, 'High',
                  np.where(risk_scores > 0.4, 'Medium', 'Low'))
    
    # 建立輸出 DataFrame
    output_df = pd.DataFrame({
        'lat': df_test_ca['Start_Lat'].values,
        'lng': df_test_ca['Start_Lng'].values,
        'timestamp': df_test_ca['Start_Time'].values,
        'actual_severity': df_test_ca['Severity'].values,
        'predicted_severity': predicted_severity,
        'risk_score': risk_scores,
        'risk_level': risk_levels,
        'hour': df_test_processed['Hour'].values,
        'day_of_week': df_test_processed['DayOfWeek'].values,
        'month': df_test_processed['Month'].values,
        'year': df_test_processed['Year'].values,
        'is_weekend': df_test_processed['IsWeekend'].values,
        'is_rush_hour': df_test_processed['IsRushHour'].values,
        'weather_category': df_test_processed['Weather_Category'].values,
        'city': df_test_ca['City'].values,
        'county': df_test_ca['County'].values
    })
    
    # 隨機抽樣以控制檔案大小（目標 100-300MB）
    target_rows = min(1000000, len(output_df))
    if len(output_df) > target_rows:
        print(f"抽樣 {target_rows:,} 筆資料以控制檔案大小...")
        output_df = output_df.sample(n=target_rows, random_state=42)
    
    return output_df


def main():
    """主程式"""
    # 載入資料
    file_path = 'us-accidents/US_Accidents_March23.csv'
    
    # 可以調整 sample_frac 來控制資料量（如果記憶體不足）
    df_all = load_all_data(file_path, sample_frac=1.0)
    
    # 按時間分割
    df_train, df_test = split_data_by_time(df_all)
    
    # 執行實驗
    results, df_test = run_experiment(df_train, df_test)
    
    # 產生 California Kepler.gl 預測資料
    kepler_df = create_kepler_predictions_ca(
        df_test, 
        results['model'], 
        results['feature_columns'],
        results['label_encoders']
    )
    
    # 儲存結果
    output_file = 'california_accidents_2022_04_to_2023_03_predictions.csv'
    kepler_df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n預測結果已儲存至: {output_file}")
    print(f"檔案大小: {file_size_mb:.1f} MB")
    
    # 顯示統計資訊
    print("\nCalifornia 預測結果統計:")
    print(f"總資料點: {len(kepler_df):,}")
    print(f"時間範圍: {kepler_df['timestamp'].min()} 到 {kepler_df['timestamp'].max()}")
    
    print(f"\n風險等級分布:")
    risk_dist = kepler_df['risk_level'].value_counts()
    for level, count in risk_dist.items():
        print(f"  {level}: {count:,} ({count/len(kepler_df)*100:.1f}%)")
    
    print(f"\n預測嚴重度分布:")
    pred_dist = kepler_df['predicted_severity'].value_counts().sort_index()
    for sev, count in pred_dist.items():
        print(f"  Severity {sev}: {count:,} ({count/len(kepler_df)*100:.1f}%)")
    
    # 比較實際與預測
    if 'actual_severity' in kepler_df.columns:
        # 整體準確率
        overall_accuracy = (kepler_df['actual_severity'] == kepler_df['predicted_severity']).mean()
        print(f"\nCalifornia 2022.04-2023.03 預測準確率: {overall_accuracy:.4f}")
        
        # 計算詳細指標
        y_true_ca = kepler_df['actual_severity'].values - 1
        y_pred_ca = kepler_df['predicted_severity'].values - 1
        
        print("\nCalifornia 預測詳細評估:")
        print(f"  準確率: {accuracy_score(y_true_ca, y_pred_ca):.4f}")
        print(f"  平衡準確率: {balanced_accuracy_score(y_true_ca, y_pred_ca):.4f}")
        print(f"  加權 F1: {f1_score(y_true_ca, y_pred_ca, average='weighted'):.4f}")
        print(f"  宏平均 F1: {f1_score(y_true_ca, y_pred_ca, average='macro'):.4f}")
        
        # 各類別表現
        print("\nCalifornia 各嚴重度等級預測準確率:")
        for severity in [1, 2, 3, 4]:
            mask = kepler_df['actual_severity'] == severity
            if mask.sum() > 0:
                class_acc = (kepler_df.loc[mask, 'actual_severity'] == 
                           kepler_df.loc[mask, 'predicted_severity']).mean()
                print(f"  Severity {severity}: {class_acc:.4f} ({mask.sum():,} 樣本)")
    
    # 顯示最終總結
    print("\n" + "="*50)
    print("實驗總結")
    print("="*50)
    print(f"訓練資料: 2016-01 到 2022-03 全美事故")
    print(f"測試資料: 2022-04 到 2023-03 全美事故")
    print(f"輸出資料: 2022-04 到 2023-03 California 事故")
    print(f"混合採樣策略: 欠採樣+過採樣")
    print(f"模型: XGBoost with GPU acceleration")
    print(f"\n最佳評估指標 (驗證集):")
    print(f"  平衡準確率: {results['metrics']['balanced_accuracy']:.4f}")
    print(f"  宏平均 F1: {results['metrics']['f1_macro']:.4f}")
    print(f"  訓練時間: {results['metrics']['train_time']:.1f} 秒")
    
    # 清理記憶體
    clean_memory()
    
    return results, kepler_df


# 執行主程式
if __name__ == "__main__":
    results, kepler_df = main()