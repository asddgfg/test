# models.py
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier


def get_regression_models() -> Dict[str, Any]:
    models = {
        # =========================================================
        # 1. Elastic Net (REGRESSION)
        # =========================================================
        "elastic_net_a001_l02": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=42)),
        ]),
        "elastic_net_a001_l05": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]),
        "elastic_net_a01_l05": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]),
        "elastic_net_a01_l08": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=10000, random_state=42)),
        ]),
        "elastic_net_a1_l05": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]),

        # =========================================================
        # 2. SVR
        # =========================================================
        "svr_rbf_c1_g01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma=0.1, epsilon=0.01)),
        ]),
        "svr_rbf_c10_g01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=0.01)),
        ]),
        "svr_rbf_c10_g001": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, gamma=0.01, epsilon=0.01)),
        ]),

        # =========================================================
        # 3. Random Forest
        # =========================================================
        "rf_d4_leaf3": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ]),
        "rf_d6_leaf5": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1)),
        ]),
        "rf_d8_leaf10": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=-1)),
        ]),

        # =========================================================
        # 4. XGBoost
        # =========================================================
        "xgb_d3_lr005": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)),
        ]),
        "xgb_d5_lr005": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)),
        ]),
        "xgb_d5_lr01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)),
        ]),
    }
    return models


def get_classification_models() -> Dict[str, Any]:
    models = {

        # =========================================================
        # 🔥 NEW: Elastic Net Logistic Regression
        # =========================================================
        "logit_enet_c01_l02": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=0.1,
                l1_ratio=0.2,
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "logit_enet_c1_l05": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=1.0,
                l1_ratio=0.5,
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "logit_enet_c10_l08": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=10.0,
                l1_ratio=0.8,
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        # =========================================================
        # SVM
        # =========================================================
        "svc_rbf_c1_g01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma=0.1, probability=True, random_state=42)),
        ]),
        "svc_rbf_c10_g01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10.0, gamma=0.1, probability=True, random_state=42)),
        ]),
        "svc_rbf_c10_g001": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10.0, gamma=0.01, probability=True, random_state=42)),
        ]),

        # =========================================================
        # Random Forest
        # =========================================================
        "rf_d4_leaf3": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ]),
        "rf_d6_leaf5": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1)),
        ]),
        "rf_d8_leaf10": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=-1)),
        ]),

        # =========================================================
        # XGBoost
        # =========================================================
        "xgbc_d3_lr005": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ]),
        "xgbc_d5_lr005": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ]),
        "xgbc_d5_lr01": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ]),
    }

    return models