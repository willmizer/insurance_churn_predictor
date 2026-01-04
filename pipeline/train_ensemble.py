import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import joblib
import os
import argparse
import sys

# configuration
# determine script directory for robust relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # one level up

DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, 'home_insurance.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Web', 'models')
OUTPUT_DIR = SCRIPT_DIR
TARGET_RECALL = 0.80
RANDOM_STATE = 42

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")

    # --- step 1: basic cleaning (from clean_data.ipynb) ---
    print("Performing basic cleaning...")
    
    # drop irrelevant columns
    cols_to_drop = ['i', 'CAMPAIGN_DESC', 'Police']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # drop rows with >50% missing
    threshold = len(df.columns) * 0.5
    df.dropna(thresh=threshold, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # fill numeric missing values
    df['MTA_FAP'] = df['MTA_FAP'].fillna(0)
    df['MTA_APRP'] = df['MTA_APRP'].fillna(0)
    df['PAYMENT_FREQUENCY'] = df['PAYMENT_FREQUENCY'].fillna(1) # default to annually if missing
  
    df['PAYMENT_FREQUENCY'] = df['PAYMENT_FREQUENCY'].fillna(2)

    # impute risk rated areas by prop type
    if 'RISK_RATED_AREA_B' in df.columns and 'PROP_TYPE' in df.columns:
        df['RISK_RATED_AREA_B'] = df.groupby('PROP_TYPE')['RISK_RATED_AREA_B'].transform(lambda x: x.fillna(x.median()))
        df['RISK_RATED_AREA_B'] = df['RISK_RATED_AREA_B'].fillna(df['RISK_RATED_AREA_B'].median())
    
    if 'RISK_RATED_AREA_C' in df.columns and 'PROP_TYPE' in df.columns:
        df['RISK_RATED_AREA_C'] = df.groupby('PROP_TYPE')['RISK_RATED_AREA_C'].transform(lambda x: x.fillna(x.median()))
        df['RISK_RATED_AREA_C'] = df['RISK_RATED_AREA_C'].fillna(df['RISK_RATED_AREA_C'].median())

    # date handling
    date_cols = ['QUOTE_DATE', 'COVER_START', 'P1_DOB', 'MTA_DATE']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # fill quote_date with cover_start
    df['QUOTE_DATE'] = df['QUOTE_DATE'].fillna(df['COVER_START'])
    
    # mta flag
    # if mta_date is missing, it implies no adjustment
    # model_prep created has_adjusted and days_since_mta
    
    # --- step 2: feature engineering (from model_prep.ipynb) ---
    print("Feature engineering...")
    
    # p1_age
    df['P1_AGE'] = (df['QUOTE_DATE'] - df['P1_DOB']).dt.days // 365
    df.loc[(df['P1_AGE'] < 16) | (df['P1_AGE'] > 100), 'P1_AGE'] = df['P1_AGE'].median()
    df['P1_AGE'] = df['P1_AGE'].fillna(df['P1_AGE'].median())

    # prop_age
    df['PROP_AGE'] = df['COVER_START'].dt.year - df['YEARBUILT']
    df.loc[(df['PROP_AGE'] < 0) | (df['PROP_AGE'] > 400), 'PROP_AGE'] = df['PROP_AGE'].median()
    df['PROP_AGE'] = df['PROP_AGE'].fillna(df['PROP_AGE'].median())

    # days_to_bind
    df['DAYS_TO_BIND'] = (df['COVER_START'] - df['QUOTE_DATE']).dt.days
    df['DAYS_TO_BIND'] = df['DAYS_TO_BIND'].clip(lower=0, upper=60).fillna(0)

    # days_since_mta & has_mta
    df['DAYS_SINCE_MTA'] = (df['QUOTE_DATE'] - df['MTA_DATE']).dt.days
    df.loc[df['DAYS_SINCE_MTA'] < 0, 'DAYS_SINCE_MTA'] = -1
    df['DAYS_SINCE_MTA'] = df['DAYS_SINCE_MTA'].fillna(-1)
    
    # has_adjusted / has_mta (modeling notebook has both)
    df['HAS_ADJUSTED'] = (df['DAYS_SINCE_MTA'] != -1).astype(int)
    df['HAS_MTA'] = df['HAS_ADJUSTED']  # duplicate for compatibility
    
    # is_standard_occupancy (derived from occ_status)
    if 'OCC_STATUS' in df.columns:
        # standard usually means owner occupied
        df['IS_STANDARD_OCCUPANCY'] = df['OCC_STATUS'].apply(lambda x: 1 if x in ['PH', 'Owner Occupied'] else 0)
        # drop original occ_status later via get_dummies or drop
        df.drop(columns=['OCC_STATUS'], inplace=True)

    # has_pt_emp
    if 'P1_PT_EMP_STATUS' in df.columns:
        df['HAS_PT_EMP'] = df['P1_PT_EMP_STATUS'].notna().astype(int)
        df.drop(columns=['P1_PT_EMP_STATUS'], inplace=True)
    else:
        df['HAS_PT_EMP'] = 0

    # marital status buckets
    if 'P1_MAR_STATUS' in df.columns:
        def bucket_mar(x):
            if x in ['M', 'P', 'C', 'B', 'Married', 'Partnered']: return 'Couple'
            if x in ['S', 'D', 'W', 'A', 'Single', 'Divorced', 'Widowed', 'Separated']: return 'Solo'
            return 'Other'
        
        df['temp_mar'] = df['P1_MAR_STATUS'].apply(bucket_mar)
        mar_dummies = pd.get_dummies(df['temp_mar'], prefix='MAR_STATUS', dtype=int)
        df = pd.concat([df, mar_dummies], axis=1)
        df.drop(columns=['P1_MAR_STATUS', 'temp_mar'], inplace=True)
        # ensure standard columns exist even if not in data
        for col in ['MAR_STATUS_Couple', 'MAR_STATUS_Solo']:
            if col not in df.columns: df[col] = 0
        # drop MAR_STATUS_Other if it exists (to match modeling notebook)
        if 'MAR_STATUS_Other' in df.columns:
            df.drop(columns=['MAR_STATUS_Other'], inplace=True)

    # sex
    if 'P1_SEX' in df.columns:
        df['P1_SEX'] = df['P1_SEX'].map({'F': 1, 'Female': 1}).fillna(0).astype(int)

    # binaries -> 1/0
    binary_cols = ['CLAIM3YEARS', 'BUS_USE', 'CLERICAL', 'AD_BUILDINGS', 'AD_CONTENTS', 
                   'CONTENTS_COVER', 'BUILDINGS_COVER', 'P1_POLICY_REFUSED', 'APPR_ALARM', 
                   'APPR_LOCKS', 'FLOODING', 'NEIGH_WATCH', 'SAFE_INSTALLED', 'SEC_DISC_REQ', 
                   'SUBSIDENCE', 'MTA_FLAG']
    # addon columns
    addon_cols = [c for c in df.columns if 'ADDON' in c]
    binary_cols.extend(addon_cols)
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({'Y': 1, 'N': 0}).fillna(0).astype(int)

    # payment method - first map values (from clean_data.ipynb), then binarize
    # PureDD, DD-Other -> 'Direct Debit' -> 1
    # NonDD -> 'Other Method' -> 0
    if 'PAYMENT_METHOD' in df.columns:
        pay_map = {
            'PureDD': 'Direct Debit',
            'DD-Other': 'Direct Debit',
            'NonDD': 'Other Method'
        }
        df['PAYMENT_METHOD'] = df['PAYMENT_METHOD'].astype(str).str.strip().map(pay_map).fillna('Other Method')
        # Now binarize: Direct Debit = 1, all others = 0
        df['PAYMENT_METHOD'] = df['PAYMENT_METHOD'].apply(lambda x: 1 if x == 'Direct Debit' else 0).astype(int)
        
    # date extraction
    df['QUOTE_MONTH'] = df['QUOTE_DATE'].dt.month
    df['QUOTE_WEEKDAY'] = df['QUOTE_DATE'].dt.dayofweek
    
    # drop date columns
    df.drop(columns=date_cols, inplace=True, errors='ignore')

    # handle p1_emp_status (target for splitting)
    # fill na with 'other'
    if 'P1_EMP_STATUS' in df.columns:
        df['P1_EMP_STATUS'] = df['P1_EMP_STATUS'].fillna('Other')
        # map codes if raw
        emp_map = {'R': 'Retired', 'E': 'Employed', 'S': 'Student', 'H': 'Houseperson', 
                   'U': 'Unemployed', 'C': 'Carer', 'I': 'Disabled', 'V': 'Voluntary', 
                   'N': 'None'}
        # apply map only if values look like keys (length 1)
        sample = df['P1_EMP_STATUS'].dropna().iloc[0] if len(df) > 0 else 'X'
        if len(str(sample)) == 1 and str(sample) in emp_map:
             df['P1_EMP_STATUS'] = df['P1_EMP_STATUS'].map(emp_map).fillna('Other')

    # target mapping: pol_status
    # lapsed/cancelled -> 1 (churn), live -> 0
    if 'POL_STATUS' in df.columns:
        df['POL_STATUS'] = df['POL_STATUS'].apply(lambda x: 0 if x == 'Live' else 1)
    
    # final cleanup of any remaining object columns
    # exclude p1_emp_status as it is used for splitting later
    obj_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'P1_EMP_STATUS']
    if len(obj_cols) > 0:
        print(f"One-hot encoding remaining object columns: {list(obj_cols)}")
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True, dtype=int)

    return df

def train_active_model(df, output_model_dir):
    print("\n--- Training Active Model (Ensemble) ---")
    
    # split features/target
    X = df.drop(columns=['POL_STATUS'])
    y = df['POL_STATUS']
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # ensemble models
    clf_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=3,
        learning_rate=0.2,
        scale_pos_weight=3,
        seed=RANDOM_STATE,
        enable_categorical=True,
        eval_metric='logloss'
    )
    
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    clf_lgbm = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.1,
        scale_pos_weight=3,
        random_state=RANDOM_STATE,
        verbosity=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('lgbm', clf_lgbm)],
        voting='soft'
    )
    
    # fit
    print("Fitting ensemble...")
    ensemble.fit(X_train, y_train)
    
    # threshold tuning
    print("Tuning threshold...")
    y_probs = ensemble.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # target recall 80%
    valid_indices = np.where(recalls >= TARGET_RECALL)[0]
    if len(valid_indices) > 0:
        optimal_idx = (np.abs(recalls - TARGET_RECALL)).argmin()
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    else:
        optimal_threshold = 0.5
        
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(classification_report(y_test, y_pred))
    
    # save
    os.makedirs(output_model_dir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(output_model_dir, 'ensemble_model.pkl'))
    with open(os.path.join(output_model_dir, 'ensemble_threshold.txt'), 'w') as f:
        f.write(str(optimal_threshold))
        
    print("Active model saved.")

def train_retired_model(df, output_model_dir):
    print("\n--- Training Retired Model (XGBoost) ---")
    
    X = df.drop(columns=['POL_STATUS'])
    y = df['POL_STATUS']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # parameters from notebook optimization
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=3,
        seed=RANDOM_STATE,
        enable_categorical=True,
        eval_metric='logloss'
    )
    
    clf.fit(X_train, y_train)
    
    # evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # save
    joblib.dump(clf, os.path.join(output_model_dir, 'xgb_retired_optimized.pkl'))
    print("Retired model saved.")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Churn Modeling Pipeline")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE, help="Path to raw CSV data")
    args = parser.parse_args()
    
    print("Starting pipeline execution...")
    
    # 1. clean & prep
    # use argument input if provided, otherwise default
    input_file = args.input
    # handle case where script is run with positional arg (legacy support)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
         input_file = sys.argv[1]

    df = load_and_clean_data(input_file)
    
    # 2. split active/retired
    print("\nSplitting Active vs Retired...")
    
    # check if p1_emp_status exists
    if 'P1_EMP_STATUS' not in df.columns:
        print("Error: P1_EMP_STATUS missing for split (or already encoded).")
        sys.exit(1)
        
    retired_df = df[df['P1_EMP_STATUS'] == 'Retired'].copy()
    active_df = df[df['P1_EMP_STATUS'] != 'Retired'].copy()
    
    print(f"Active rows: {len(active_df)}")
    print(f"Retired rows: {len(retired_df)}")
    
    # 3. final per-dataset prep
    
    # active: encode remaining p1_emp_status values (employed, etc)
    # we use get_dummies. 
    active_df = pd.get_dummies(active_df, columns=['P1_EMP_STATUS'], prefix='EMP_STATUS', dtype=int)
    
    # retired: drop p1_emp_status as it is always 'retired'
    retired_df.drop(columns=['P1_EMP_STATUS'], inplace=True)
    
    # 4. save csvs
    print(f"\nSaving cleaned datasets to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    active_csv = os.path.join(OUTPUT_DIR, 'active_clean.csv')
    retired_csv = os.path.join(OUTPUT_DIR, 'retired_clean.csv')
    
    active_df.to_csv(active_csv, index=False)
    retired_df.to_csv(retired_csv, index=False)
    print(f"- {active_csv}")
    print(f"- {retired_csv}")
    
    # 5. train models
    train_active_model(active_df, MODELS_DIR)
    train_retired_model(retired_df, MODELS_DIR)
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
