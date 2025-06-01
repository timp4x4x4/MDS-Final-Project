# ===========================
# Cell 1: 導入套件和設定
# ===========================
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc  # 垃圾回收
warnings.filterwarnings('ignore')

# 基本套件
import os
import time
import joblib
import json
from collections import Counter

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, balanced_accuracy_score, 
    cohen_kappa_score, make_scorer
)
from sklearn.utils.class_weight import compute_class_weight

# 處理不平衡資料
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

# Boosting模型
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F 

print("環境檢查:")
print(f"PyTorch: {torch.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"LightGBM: {lgb.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===========================
# Cell 2: 記憶體優化函數
# ===========================

def reduce_memory_usage(df, verbose=True):
    """
    通過改變數據類型來減少DataFrame的記憶體使用
    參考自Kaggle的記憶體優化技術
    """
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

# ===========================
# Cell 3: 載入資料（優化版）
# ===========================

def load_data_optimized(file_path, sample_frac=None, chunksize=None):
    """
    優化的資料載入，支援採樣和分塊讀取
    """
    print(f"載入資料: {file_path}")
    
    # 先讀取一小部分來了解資料
    sample_df = pd.read_csv(file_path, nrows=5)
    print("資料欄位預覽:")
    print(sample_df.columns.tolist())
    
    # 定義需要的欄位（排除不需要的文字欄位以節省記憶體）
    # 根據其他Kaggle notebook的經驗，這些是最重要的欄位
    important_cols = [
        'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
        'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
        'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset', 'State',
        'Side', 'Weather_Timestamp'
    ]
    
    # 過濾存在的欄位
    existing_cols = [col for col in important_cols if col in sample_df.columns]
    
    # 定義數據類型以減少記憶體
    dtype_dict = {
        'Severity': 'int8',
        'Distance(mi)': 'float32',
        'Temperature(F)': 'float32',
        'Humidity(%)': 'float32',
        'Pressure(in)': 'float32',
        'Visibility(mi)': 'float32',
        'Wind_Speed(mph)': 'float32',
        'Precipitation(in)': 'float32',
        'Amenity': 'bool',
        'Bump': 'bool',
        'Crossing': 'bool',
        'Give_Way': 'bool',
        'Junction': 'bool',
        'No_Exit': 'bool',
        'Railway': 'bool',
        'Roundabout': 'bool',
        'Station': 'bool',
        'Stop': 'bool',
        'Traffic_Calming': 'bool',
        'Traffic_Signal': 'bool'
    }
    
    # 載入資料
    if sample_frac:
        # 隨機採樣
        print(f"載入 {sample_frac*100}% 的資料...")
        df = pd.read_csv(file_path, usecols=existing_cols, dtype=dtype_dict)
        df = df.sample(frac=sample_frac, random_state=42)
    elif chunksize:
        # 分塊載入
        print(f"分塊載入，每塊 {chunksize} 行...")
        chunks = []
        for chunk in pd.read_csv(file_path, usecols=existing_cols, 
                                dtype=dtype_dict, chunksize=chunksize):
            chunks.append(chunk)
            if len(chunks) * chunksize >= 1000000:  # 限制在100萬行
                break
        df = pd.concat(chunks, ignore_index=True)
    else:
        # 完整載入
        df = pd.read_csv(file_path, usecols=existing_cols, dtype=dtype_dict)
    
    print(f"載入資料大小: {df.shape}")
    print(f"記憶體使用: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # 顯示目標變數分布
    print("\n目標變數分布:")
    severity_counts = df['Severity'].value_counts().sort_index()
    for sev, count in severity_counts.items():
        print(f"Severity {sev}: {count:,} ({count/len(df)*100:.2f}%)")
    
    return df

# 執行載入（建議先用小樣本測試）
file_path = 'us-accidents/US_Accidents_March23.csv'

# 選項1: 使用部分資料（推薦用於測試）
# df = load_data_optimized(file_path, sample_frac=0.1)  # 10%資料

# 選項2: 分塊載入
# df = load_data_optimized(file_path, chunksize=500000)  # 每次50萬行

# 選項3: 完整載入（需要大量記憶體）
df = load_data_optimized(file_path)

# ===========================
# Cell 4: 日期時間處理（優化版）
# ===========================

def process_datetime_features(df):
    """處理日期時間特徵"""
    print("\n處理日期時間特徵...")
    
    # 轉換日期時間
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # 計算持續時間
    df['Duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    # 過濾異常值（使用更寬鬆的範圍）
    df = df[(df['Duration_minutes'] > 0) & (df['Duration_minutes'] < 1440*7)]  # 小於7天
    
    # 移除日期時間為空的記錄
    df = df.dropna(subset=['Start_Time'])
    
    # 提取時間特徵
    df['Hour'] = df['Start_Time'].dt.hour.astype('int8')
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek.astype('int8')
    df['Month'] = df['Start_Time'].dt.month.astype('int8')
    df['Year'] = df['Start_Time'].dt.year.astype('int16')
    
    # 衍生特徵
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype('int8')
    df['IsRushHour'] = df['Hour'].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    ).astype('int8')
    
    # 時段分類
    df['TimeOfDay'] = pd.cut(df['Hour'], 
                            bins=[-1, 6, 12, 18, 24], 
                            labels=[0, 1, 2, 3]).astype('int8')  # 轉換為數值
    
    # 季節
    df['Season'] = pd.cut(df['Month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=[0, 1, 2, 3]).astype('int8')  # 轉換為數值
    
    # 刪除原始時間欄位以節省記憶體
    df = df.drop(['Start_Time', 'End_Time', 'Weather_Timestamp'], axis=1, errors='ignore')
    
    print(f"處理後大小: {df.shape}")
    clean_memory()
    
    return df

df = process_datetime_features(df)

# ===========================
# Cell 5: 天氣特徵處理
# ===========================

def process_weather_features(df):
    """處理天氣相關特徵"""
    print("\n處理天氣特徵...")
    
    if 'Weather_Condition' in df.columns:
        # 簡化天氣分類
        def categorize_weather(condition):
            if pd.isna(condition):
                return 0  # Unknown
            condition = str(condition).lower()
            if any(word in condition for word in ['clear', 'fair']):
                return 1  # Clear
            elif any(word in condition for word in ['cloud', 'overcast']):
                return 2  # Cloudy
            elif any(word in condition for word in ['rain', 'drizzle']):
                return 3  # Rain
            elif any(word in condition for word in ['snow', 'sleet']):
                return 4  # Snow
            elif any(word in condition for word in ['fog', 'mist']):
                return 5  # Fog
            elif any(word in condition for word in ['storm', 'thunder']):
                return 6  # Storm
            else:
                return 7  # Other
        
        df['Weather_Category'] = df['Weather_Condition'].apply(categorize_weather).astype('int8')
        df = df.drop('Weather_Condition', axis=1)
    
    # 處理其他天氣數值特徵的缺失值
    weather_numeric_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                           'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    for col in weather_numeric_cols:
        if col in df.columns:
            # 使用中位數填充
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    clean_memory()
    return df

df = process_weather_features(df)

# ===========================
# Cell 6: 處理缺失值和編碼類別變數
# ===========================

def handle_missing_and_encode(df):
    """處理缺失值並編碼類別變數"""
    print("\n處理缺失值和編碼...")
    
    # 刪除缺失值過多的欄位
    missing_pct = df.isnull().sum() / len(df)
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    
    # 保留Severity
    if 'Severity' in high_missing_cols:
        high_missing_cols.remove('Severity')
    
    df = df.drop(columns=high_missing_cols, errors='ignore')
    print(f"刪除高缺失率欄位: {len(high_missing_cols)}")
    
    # 對類別變數進行標籤編碼
    categorical_cols = ['State', 'Side', 'Sunrise_Sunset']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # 填充缺失值
            df[col] = df[col].fillna('Unknown')
            # 編碼
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # 填充數值型缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Severity':
            df[col] = df[col].fillna(df[col].median())
    
    # 確保布林型欄位是整數
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype('int8')
    
    print(f"處理後資料大小: {df.shape}")
    print(f"剩餘缺失值: {df.isnull().sum().sum()}")
    
    clean_memory()
    return df, label_encoders

df, label_encoders = handle_missing_and_encode(df)

# ===========================
# Cell 7: 特徵選擇和準備最終數據
# ===========================

def prepare_final_data(df):
    """準備最終的訓練數據"""
    print("\n準備最終數據...")
    
    # 刪除任何仍有缺失值的行
    df = df.dropna()
    
    # 確保Severity是正確的值
    df = df[df['Severity'].isin([1, 2, 3, 4])]
    
    # 根據Kaggle上的建議，考慮合併Severity 1和2
    # 因為Severity 1的樣本太少
    print("\n原始類別分布:")
    print(df['Severity'].value_counts().sort_index())
    
    # 選項：合併類別（可選）
    # df['Severity'] = df['Severity'].replace({1: 2})
    
    # 分離特徵和目標
    feature_cols = [col for col in df.columns if col != 'Severity']
    X = df[feature_cols].values
    y = df['Severity'].values - 1  # 轉換為0-3
    
    print(f"\n最終數據大小: X={X.shape}, y={y.shape}")
    print("最終類別分布:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  類別 {cls} (Severity {cls+1}): {cnt:,} ({cnt/len(y)*100:.2f}%)")
    
    return X, y, feature_cols

X, y, feature_names = prepare_final_data(df)

# ===========================
# Cell 8: 資料分割
# ===========================

# 分層分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練集: {X_train.shape}")
print(f"測試集: {X_test.shape}")

# 計算類別權重
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_train), 
                                   y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print("\n類別權重:")
for cls, weight in class_weight_dict.items():
    print(f"  類別 {cls}: {weight:.4f}")

# ===========================
# Cell 9: 處理不平衡 - 混合採樣策略
# ===========================

def apply_mixed_sampling(X_train, y_train, strategy='mixed'):
    """
    應用混合採樣策略
    參考Kaggle最佳實踐：結合過採樣和欠採樣
    """
    print(f"\n應用採樣策略: {strategy}")
    
    if strategy == 'none':
        return X_train, y_train
    
    # 計算各類別數量
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("原始分布:", class_counts)
    
    if strategy == 'mixed':
        # 混合策略：對多數類欠採樣，對少數類過採樣
        # 目標：讓所有類別接近中位數
        median_count = int(np.median(counts))
        target_count = int(median_count * 1.5)  # 目標數量設為中位數的1.5倍
        
        # 第一步：欠採樣 - 只對超過目標數量的類別進行欠採樣
        undersample_strategy = {}
        for cls, cnt in class_counts.items():
            if cnt > target_count:
                undersample_strategy[cls] = target_count
            else:
                undersample_strategy[cls] = cnt  # 保持原樣
        
        if len(undersample_strategy) > 0 and any(cnt < class_counts[cls] for cls, cnt in undersample_strategy.items()):
            rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
            X_temp, y_temp = rus.fit_resample(X_train, y_train)
        else:
            X_temp, y_temp = X_train, y_train
        
        # 第二步：過採樣 - 只對少於目標數量的類別進行過採樣
        temp_unique, temp_counts = np.unique(y_temp, return_counts=True)
        temp_class_counts = dict(zip(temp_unique, temp_counts))
        
        oversample_strategy = {}
        for cls, cnt in temp_class_counts.items():
            if cnt < target_count:
                oversample_strategy[cls] = target_count
            else:
                oversample_strategy[cls] = cnt  # 保持原樣
        
        if len(oversample_strategy) > 0 and any(cnt > temp_class_counts[cls] for cls, cnt in oversample_strategy.items()):
            ros = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X_temp, y_temp)
        else:
            X_resampled, y_resampled = X_temp, y_temp
            
    elif strategy == 'smote':
        # SMOTE策略：只過採樣到最多類別的50%
        max_count = max(counts)
        target_count = int(max_count * 0.5)
        
        # 確保目標數量不小於當前數量
        sampling_strategy = {}
        for cls, cnt in class_counts.items():
            if cnt < target_count:
                sampling_strategy[cls] = target_count
            else:
                sampling_strategy[cls] = cnt
        
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    elif strategy == 'undersample_only':
        # 只欠採樣到最少類別的2倍
        min_count = min(counts)
        target_count = min_count * 2
        
        sampling_strategy = {}
        for cls, cnt in class_counts.items():
            sampling_strategy[cls] = min(cnt, target_count)
        
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    # 顯示新分布
    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    new_distribution = dict(zip(unique_new, counts_new))
    print("採樣後分布:", new_distribution)
    
    # 顯示變化
    print("\n採樣變化:")
    for cls in range(4):
        original = class_counts.get(cls, 0)
        new = new_distribution.get(cls, 0)
        change = ((new - original) / original * 100) if original > 0 else 0
        print(f"  類別 {cls}: {original:,} → {new:,} ({change:+.1f}%)")
    
    return X_resampled, y_resampled

# 應用混合採樣
# 可以嘗試不同策略
X_train_balanced, y_train_balanced = apply_mixed_sampling(X_train, y_train, 'mixed')

# 如果混合策略還是有問題，可以嘗試其他策略：
# X_train_balanced, y_train_balanced = apply_mixed_sampling(X_train, y_train, 'undersample_only')
# 或者不進行採樣：
# X_train_balanced, y_train_balanced = X_train, y_train

# ===========================
# Cell 10: LightGBM模型（優化版）
# ===========================

def train_lightgbm_optimized(X_train, X_test, y_train, y_test, class_weights):
    """訓練優化的LightGBM模型"""
    print("\n訓練 LightGBM (優化版)...")
    
    # 創建數據集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 參數設置（基於Kaggle最佳實踐）
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': -1,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'min_split_gain': 0.02,
        'class_weight': 'balanced',
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 訓練
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    train_time = time.time() - start_time
    
    # 預測
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n訓練時間: {train_time:.2f} 秒")
    print(f"最佳迭代次數: {model.best_iteration}")
    print(f"準確率: {accuracy:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"平衡準確率: {balanced_acc:.4f}")
    
    # 詳細報告
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f'Severity {i+1}' for i in range(4)]))
    
    return model, accuracy, f1, balanced_acc

# 訓練模型
lgb_model, lgb_acc, lgb_f1, lgb_balanced_acc = train_lightgbm_optimized(
    X_train_balanced, X_test, y_train_balanced, y_test, class_weight_dict
)

# ===========================
# Cell 11: XGBoost模型（穩定版）
# ===========================

def train_xgboost_stable(X_train, X_test, y_train, y_test, use_sample_weight=True):
    """穩定版XGBoost"""
    print("\n訓練 XGBoost (穩定版)...")
    
    # 使用原始的類別權重，但不要太極端
    if use_sample_weight:
        # 溫和的類別權重
        unique, counts = np.unique(y_train, return_counts=True)
        weight_dict = {}
        max_count = max(counts)
        for cls, count in zip(unique, counts):
            # 權重不超過10倍
            weight_dict[cls] = min(max_count / count, 10.0)
        
        sample_weights = np.array([weight_dict[y] for y in y_train])
    else:
        sample_weights = None
    
    # XGBoost參數
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,  # 增加以防止過擬合
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    # 訓練
    model = xgb.XGBClassifier(**params)
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        # early_stopping_rounds=50,
        verbose=100
    )
    train_time = time.time() - start_time
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n訓練時間: {train_time:.2f} 秒")
    print(f"準確率: {accuracy:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"平衡準確率: {balanced_acc:.4f}")
    
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f'Severity {i+1}' for i in range(4)]))
    
    return model, accuracy, f1, balanced_acc

# 執行訓練
xgb_model, xgb_acc, xgb_f1, xgb_balanced_acc = train_xgboost_stable(
    X_train_balanced, X_test, y_train_balanced, y_test
)

# ===========================
# Cell 12: CatBoost模型
# ===========================

def train_catboost_optimized(X_train, X_test, y_train, y_test):
    """訓練優化的CatBoost模型"""
    print("\n訓練 CatBoost (優化版)...")
    
    # CatBoost參數
    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=1,
        od_type='Iter',
        od_wait=50,
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        devices='0',
        random_state=42,
        verbose=100
    )
    
    # 訓練
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        plot=False
    )
    
    train_time = time.time() - start_time
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n訓練時間: {train_time:.2f} 秒")
    print(f"準確率: {accuracy:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"平衡準確率: {balanced_acc:.4f}")
    
    return model, accuracy, f1, balanced_acc

# 訓練CatBoost
cat_model, cat_acc, cat_f1, cat_balanced_acc = train_catboost_optimized(
    X_train_balanced, X_test, y_train_balanced, y_test
)

# ===========================
# Cell 13: Balanced Random Forest
# ===========================

def train_balanced_rf(X_train, X_test, y_train, y_test):
    """訓練Balanced Random Forest"""
    print("\n訓練 Balanced Random Forest...")
    
    model = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n訓練時間: {train_time:.2f} 秒")
    print(f"OOB分數: {model.oob_score_:.4f}")
    print(f"準確率: {accuracy:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"平衡準確率: {balanced_acc:.4f}")
    
    # 詳細報告
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f'Severity {i+1}' for i in range(4)]))
    
    return model, accuracy, f1, balanced_acc

# 訓練Balanced RF
brf_model, brf_acc, brf_f1, brf_balanced_acc = train_balanced_rf(
    X_train, X_test, y_train, y_test  # 使用原始數據，因為模型內部會平衡
)

# ===========================
# 改進的深度學習模型（替換 Cell 14）
# ===========================

class ImprovedNN(nn.Module):
    """改進的神經網路 - 加入更多技巧"""
    def __init__(self, input_size, num_classes=4):
        super(ImprovedNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, num_classes)
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

def train_improved_nn(X_train, X_test, y_train, y_test, epochs=100):  # epochs在這裡
    """訓練改進的深度學習模型"""
    print("\n訓練改進的神經網路...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 標準化
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 轉換為張量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    
    # 創建模型 - 不傳入epochs
    model = ImprovedNN(X_train.shape[1]).to(device)
    
    # 損失函數
    class_weights_tensor = torch.FloatTensor(list(class_weight_dict.values())).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練循環
    start_time = time.time()
    best_balanced_acc = 0
    
    for epoch in range(epochs):  # epochs在這裡使用
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 每10個epoch評估一次
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                _, predicted = torch.max(val_outputs, 1)
                val_balanced_acc = balanced_accuracy_score(y_test, predicted.cpu().numpy())
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
                  f"Balanced Acc: {val_balanced_acc:.4f}")
            
            if val_balanced_acc > best_balanced_acc:
                best_balanced_acc = val_balanced_acc
                best_model_state = model.state_dict()
    
    # 載入最佳模型
    if best_balanced_acc > 0:
        model.load_state_dict(best_model_state)
    
    # 最終評估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
    
    train_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n訓練時間: {train_time:.2f} 秒")
    print(f"準確率: {accuracy:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"平衡準確率: {balanced_acc:.4f}")
    
    return model, scaler, accuracy, f1, balanced_acc

# 執行訓練
nn_model, nn_scaler, nn_acc, nn_f1, nn_balanced_acc = train_improved_nn(
    X_train_balanced, X_test, y_train_balanced, y_test, epochs=50
)

# ===========================
# Cell 15: 模型比較和集成
# ===========================

# 收集所有結果
results = {
    'LightGBM': {'accuracy': lgb_acc, 'f1': lgb_f1, 'balanced_acc': lgb_balanced_acc},
    'XGBoost': {'accuracy': xgb_acc, 'f1': xgb_f1, 'balanced_acc': xgb_balanced_acc},
    'CatBoost': {'accuracy': cat_acc, 'f1': cat_f1, 'balanced_acc': cat_balanced_acc},
    'Balanced_RF': {'accuracy': brf_acc, 'f1': brf_f1, 'balanced_acc': brf_balanced_acc},
    'Neural_Network': {'accuracy': nn_acc, 'f1': nn_f1, 'balanced_acc': nn_balanced_acc}
}

print("\n" + "="*70)
print("模型性能比較")
print("="*70)
print(f"{'模型':<20} {'準確率':<10} {'F1分數':<10} {'平衡準確率':<10}")
print("-"*70)

for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['balanced_acc'], reverse=True):
    print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['balanced_acc']:<10.4f}")

# 找出最佳模型
best_model_name = max(results.items(), key=lambda x: x[1]['balanced_acc'])[0]
print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   平衡準確率: {results[best_model_name]['balanced_acc']:.4f}")

# ===========================
# Cell 16: 保存模型和結果（修正版）
# ===========================

def save_models_and_results(models, results, feature_names, label_encoders):
    """保存所有模型和結果"""
    output_dir = './model_output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model_dict = {
        'LightGBM': lgb_model,
        'XGBoost': xgb_model,
        'CatBoost': cat_model,
        'Balanced_RF': brf_model
    }
    
    for name, model in model_dict.items():
        if name == 'LightGBM':
            model.save_model(f'{output_dir}{name.lower()}_model.txt')
        else:
            joblib.dump(model, f'{output_dir}{name.lower()}_model.pkl')
    
    # 保存神經網路
    torch.save(nn_model.state_dict(), f'{output_dir}neural_network_model.pth')
    joblib.dump(nn_scaler, f'{output_dir}nn_scaler.pkl')
    
    # 保存特徵名稱和編碼器
    joblib.dump(feature_names, f'{output_dir}feature_names.pkl')
    joblib.dump(label_encoders, f'{output_dir}label_encoders.pkl')
    
    # 保存結果
    import json
    with open(f'{output_dir}results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存訓練信息 - 修正：將numpy類型轉換為Python原生類型
    train_info = {
        'train_size': int(len(X_train)),  # 轉換為int
        'test_size': int(len(X_test)),    # 轉換為int
        'n_features': int(len(feature_names)),  # 轉換為int
        'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},  # 轉換鍵值
        'best_model': best_model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}train_info.json', 'w') as f:
        json.dump(train_info, f, indent=4)
    
    print(f"\n✅ 所有模型和結果已保存至: {output_dir}")

# 保存
save_models_and_results(
    {'nn_model': nn_model, 'nn_scaler': nn_scaler},
    results,
    feature_names,
    label_encoders
)