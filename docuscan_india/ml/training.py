import os
import sys
import pickle

# Ensure the root directory of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from ml.feature_engineering import FEATURE_NAMES, FeatureEngineering

from utils.logger import get_logger

logger = get_logger("training")

def generate_mock_dataset(n_samples: int = 1000):
    """
    Generates a synthetic dataset containing features corresponding to:
      1. Genuine documents (high quality, valid details, untampered)
      2. Tampered documents (EXIF edit tags, overlapping fields, checksum failures)
      3. Blurry / Low-Quality documents (poor OCR confidence, low Laplacian variance)
      4. Mismatched documents (classification conflicts, wrong keywords/regex)
    """
    np.random.seed(42)
    data = []

    # Target labels to train:
    # - target_ocr: OCR confidence is high/reliable
    # - target_class: Classification is correct/high-confidence
    # - target_extract: Extraction is complete/reliable
    # - target_fraud: Document shows signs of tampering/fraud (1 = Fraud, 0 = Safe)
    # - target_authentic: Overall document is genuine (1 = Genuine, 0 = Suspicious/Bad)
    
    labels_ocr = []
    labels_class = []
    labels_extract = []
    labels_fraud = []
    labels_authentic = []

    for i in range(n_samples):
        # Determine document category: 0=Genuine, 1=Tampered, 2=Blurry, 3=Mismatched
        cat = np.random.choice([0, 1, 2, 3], p=[0.5, 0.2, 0.2, 0.1])
        
        # Base defaults
        ocr_mean = np.random.uniform(0.15, 0.99)
        ocr_min = float(np.clip(ocr_mean - np.random.uniform(0.01, min(0.3, ocr_mean)), 0.0, 1.0))
        ocr_max = float(np.clip(ocr_mean + np.random.uniform(0.01, min(0.3, 1.0 - ocr_mean)), 0.0, 1.0))
        ocr_std = np.random.uniform(0.01, min(0.15, (ocr_max - ocr_min) / 2.0 + 0.01))
        pct_low = float(np.clip(1.0 - ocr_mean + np.random.normal(0, 0.1), 0.0, 1.0))
        ocr_median = float(np.clip(ocr_mean + np.random.uniform(-0.05, 0.05), 0.0, 1.0))
        
        word_cnt = np.random.randint(15, 60)
        char_cnt = word_cnt * np.random.randint(4, 7)
        avg_w_len = np.random.uniform(4.5, 6.5)

        # Image quality (high values = good quality)
        if cat == 2:  # Blurry / Low Quality
            blur_score = np.random.uniform(10.0, 500.0)
            ocr_mean = np.random.uniform(0.15, 0.65)
            ocr_min = float(np.clip(ocr_mean - np.random.uniform(0.01, min(0.3, ocr_mean)), 0.0, 1.0))
            ocr_max = float(np.clip(ocr_mean + np.random.uniform(0.01, min(0.3, 1.0 - ocr_mean)), 0.0, 1.0))
            ocr_std = np.random.uniform(0.01, min(0.15, (ocr_max - ocr_min) / 2.0 + 0.01))
            pct_low = float(np.clip(1.0 - ocr_mean + np.random.normal(0, 0.1), 0.0, 1.0))
            ocr_median = float(np.clip(ocr_mean + np.random.uniform(-0.05, 0.05), 0.0, 1.0))
        elif cat == 0:  # Genuine
            blur_score = np.random.uniform(500.0, 3000.0)
            ocr_mean = np.random.uniform(0.60, 0.99)
            ocr_min = float(np.clip(ocr_mean - np.random.uniform(0.01, min(0.3, ocr_mean)), 0.0, 1.0))
            ocr_max = float(np.clip(ocr_mean + np.random.uniform(0.01, min(0.3, 1.0 - ocr_mean)), 0.0, 1.0))
            ocr_std = np.random.uniform(0.01, min(0.15, (ocr_max - ocr_min) / 2.0 + 0.01))
            pct_low = float(np.clip(1.0 - ocr_mean + np.random.normal(0, 0.1), 0.0, 1.0))
            ocr_median = float(np.clip(ocr_mean + np.random.uniform(-0.05, 0.05), 0.0, 1.0))
        else:
            blur_score = np.random.uniform(300.0, 2000.0)
            ocr_mean = np.random.uniform(0.40, 0.95)
            ocr_min = float(np.clip(ocr_mean - np.random.uniform(0.01, min(0.3, ocr_mean)), 0.0, 1.0))
            ocr_max = float(np.clip(ocr_mean + np.random.uniform(0.01, min(0.3, 1.0 - ocr_mean)), 0.0, 1.0))
            ocr_std = np.random.uniform(0.01, min(0.15, (ocr_max - ocr_min) / 2.0 + 0.01))
            pct_low = float(np.clip(1.0 - ocr_mean + np.random.normal(0, 0.1), 0.0, 1.0))
            ocr_median = float(np.clip(ocr_mean + np.random.uniform(-0.05, 0.05), 0.0, 1.0))

        noise_score = np.random.uniform(40.0, 75.0) if cat != 2 else np.random.uniform(5.0, 25.0)
        edge_density = np.random.uniform(0.08, 0.18)

        bbox_h_std = np.random.uniform(2.0, 5.0)
        bbox_w_std = np.random.uniform(5.0, 15.0)
        bbox_area = np.random.uniform(0.10, 0.35)

        # Classifiers
        doc_type_idx = np.random.randint(0, 4) # 0=Aadhaar, 1=PAN, 2=Passport, 3=DL
        
        # Init scores
        kw_scores = [0.0] * 4
        rg_scores = [0.0] * 4
        ly_scores = [0.0] * 4
        
        kw_scores[doc_type_idx] = np.random.uniform(0.75, 0.95)
        rg_scores[doc_type_idx] = np.random.uniform(0.75, 0.98)
        ly_scores[doc_type_idx] = np.random.uniform(0.80, 0.95)

        # Validations
        val_fails = 0.0
        val_warns = 0.0
        checksum_fail = 0.0
        missing_fields = 0.0
        overlaps = 0.0
        layout_anoms = 0.0
        exif_edit = 0.0
        exif_time = 0.0

        # Regex match counts
        reg_a = 1.0 if doc_type_idx == 0 else 0.0
        reg_p = 1.0 if doc_type_idx == 1 else 0.0
        reg_pass = 1.0 if doc_type_idx == 2 else 0.0
        reg_dl = 1.0 if doc_type_idx == 3 else 0.0

        # Adjust based on document category
        if cat == 0:  # Genuine
            valid_words_ratio = np.random.uniform(0.85, 1.0)
            alphanumeric_ratio = np.random.uniform(0.75, 0.90)
            dictionary_match_ratio = np.random.uniform(0.20, 0.50)
            field_extraction_success_rate = np.random.uniform(0.85, 1.0)
            val_fails = 0.0
            val_warns = float(np.random.choice([0, 1], p=[0.8, 0.2]))
            checksum_fail = 0.0
            missing_fields = 0.0
            
            t_class, t_extract, t_fraud, t_auth = 1, 1, 0, 1

        elif cat == 1:  # Tampered
            valid_words_ratio = np.random.uniform(0.70, 0.95)
            alphanumeric_ratio = np.random.uniform(0.70, 0.88)
            dictionary_match_ratio = np.random.uniform(0.15, 0.40)
            field_extraction_success_rate = np.random.uniform(0.50, 0.85)
            exif_edit = float(np.random.choice([0, 1], p=[0.3, 0.7]))
            exif_time = float(np.random.choice([0, 1], p=[0.4, 0.6]))
            overlaps = float(np.random.randint(1, 3))
            layout_anoms = float(np.random.choice([0, 1], p=[0.5, 0.5]))
            checksum_fail = float(np.random.choice([0, 1], p=[0.2, 0.8]))
            val_fails = float(np.random.randint(1, 4))
            missing_fields = float(np.random.randint(0, 2))
            
            t_class, t_extract, t_fraud, t_auth = 1, 0, 1, 0

        elif cat == 2:  # Blurry / Low Quality
            valid_words_ratio = np.random.uniform(0.15, 0.50)
            alphanumeric_ratio = np.random.uniform(0.30, 0.65)
            dictionary_match_ratio = np.random.uniform(0.02, 0.12)
            field_extraction_success_rate = np.random.uniform(0.0, 0.30)
            missing_fields = float(np.random.randint(1, 4))
            val_fails = float(np.random.randint(0, 2))
            
            t_class, t_extract, t_fraud, t_auth = 0, 0, 0, 0

        else:  # Mismatched Document
            valid_words_ratio = np.random.uniform(0.75, 0.95)
            alphanumeric_ratio = np.random.uniform(0.70, 0.90)
            dictionary_match_ratio = np.random.uniform(0.05, 0.20)
            field_extraction_success_rate = np.random.uniform(0.10, 0.50)
            kw_scores = [0.0] * 4
            rg_scores = [0.0] * 4
            kw_scores[np.random.randint(0, 4)] = 0.8
            rg_scores[np.random.randint(0, 4)] = 0.9
            val_fails = float(np.random.randint(1, 3))
            missing_fields = float(np.random.randint(2, 4))
            
            t_class, t_extract, t_fraud, t_auth = 0, 0, 0, 0

        text_density = float(char_cnt) / 1000000.0

        # Bernoulli probability for OCR confidence label based on features
        scaled_blur = min(1.0, blur_score / 1500.0)
        blur_penalty = 1.5 * max(0.0, (400.0 - blur_score) / 400.0)
        height_std_penalty = 1.0 * max(0.0, bbox_h_std - 3.5)
        
        logit_ocr = (
            4.0 * ocr_mean 
            + 0.5 * scaled_blur 
            + 0.5 * dictionary_match_ratio 
            + 0.5 * valid_words_ratio
            - 3.0 * pct_low 
            - blur_penalty 
            - height_std_penalty 
            - 1.0
        )
        prob_ocr = 1.0 / (1.0 + np.exp(-logit_ocr))
        t_ocr = int(np.random.binomial(1, prob_ocr))

        # Build row dictionary in stable order
        row = {}
        row["ocr_mean_conf"] = ocr_mean
        row["ocr_min_conf"] = ocr_min
        row["ocr_max_conf"] = ocr_max
        row["ocr_std_conf"] = ocr_std
        row["ocr_pct_low_conf"] = pct_low
        row["ocr_median_conf"] = ocr_median
        row["word_count"] = float(word_cnt)
        row["char_count"] = float(char_cnt)
        row["avg_word_len"] = avg_w_len
        row["valid_words_ratio"] = valid_words_ratio
        row["alphanumeric_ratio"] = alphanumeric_ratio
        row["dictionary_match_ratio"] = dictionary_match_ratio
        row["text_density"] = text_density
        row["kw_score_aadhaar"] = kw_scores[0]
        row["kw_score_pan"] = kw_scores[1]
        row["kw_score_passport"] = kw_scores[2]
        row["kw_score_dl"] = kw_scores[3]
        row["rg_score_aadhaar"] = rg_scores[0]
        row["rg_score_pan"] = rg_scores[1]
        row["rg_score_passport"] = rg_scores[2]
        row["rg_score_dl"] = rg_scores[3]
        row["ly_score_aadhaar"] = ly_scores[0]
        row["ly_score_pan"] = ly_scores[1]
        row["ly_score_passport"] = ly_scores[2]
        row["ly_score_dl"] = ly_scores[3]
        row["blur_score"] = blur_score
        row["noise_score"] = noise_score
        row["edge_density"] = edge_density
        row["bbox_height_std"] = bbox_h_std
        row["bbox_width_std"] = bbox_w_std
        row["bbox_area_ratio"] = bbox_area
        row["validation_fails"] = val_fails
        row["validation_warns"] = val_warns
        row["checksum_failed"] = checksum_fail
        row["missing_fields_count"] = missing_fields
        row["field_extraction_success_rate"] = field_extraction_success_rate
        row["overlapping_fields_count"] = overlaps
        row["layout_anomaly_count"] = layout_anoms
        row["exif_editor_detected"] = exif_edit
        row["exif_timestamp_mismatch"] = exif_time
        row["regex_match_aadhaar"] = reg_a
        row["regex_match_pan"] = reg_p
        row["regex_match_passport"] = reg_pass
        row["regex_match_dl"] = reg_dl

        data.append(row)
        
        labels_ocr.append(t_ocr)
        labels_class.append(t_class)
        labels_extract.append(t_extract)
        labels_fraud.append(t_fraud)
        labels_authentic.append(t_auth)

    df = pd.DataFrame(data)
    # Ensure columns match FEATURE_NAMES order
    df = df[FEATURE_NAMES]

    return df, {
        "ocr": np.array(labels_ocr),
        "classification": np.array(labels_class),
        "extraction": np.array(labels_extract),
        "fraud": np.array(labels_fraud),
        "authenticity": np.array(labels_authentic)
    }

def train_and_save_models():
    """Trains LightGBM models for scoring and saves them to local files."""
    logger.info("Generating training data...")
    X, ys = generate_mock_dataset(5000)

    trained_models = {}
    calibration_info = {}
    optimal_thresholds = {}

    from sklearn.metrics import f1_score

    for target_name, y in ys.items():
        logger.info(f"Training calibrated model for target: {target_name}...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Configure LightGBM classifier with stable, robust parameters for small structured datasets
        base_estimator = LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            verbosity=-1
        )
        
        # Wrap with CalibratedClassifierCV to get probability calibration
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=3
        )
        
        calibrated_clf.fit(X_train, y_train)
        
        # Score the model
        acc = calibrated_clf.score(X_test, y_test)
        logger.info(f"Model '{target_name}' test accuracy: {acc * 100:.2f}%")
        
        trained_models[target_name] = calibrated_clf
        
        # Extract calibration metrics/distributions for audit/debugging
        probs = calibrated_clf.predict_proba(X_test)[:, 1]
        calibration_info[target_name] = {
            "mean_prob": float(np.mean(probs)),
            "std_prob": float(np.std(probs)),
            "min_prob": float(np.min(probs)),
            "max_prob": float(np.max(probs))
        }

        # Find optimal threshold to maximize F1 score on test set
        threshold_scores = []
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs >= thresh).astype(int)
            score = f1_score(y_test, preds, zero_division=0)
            threshold_scores.append((thresh, score))
        
        max_f1 = max(s[1] for s in threshold_scores)
        best_thresholds = [s[0] for s in threshold_scores if abs(s[1] - max_f1) < 1e-6]
        best_thresh = float(np.median(best_thresholds))

        optimal_thresholds[target_name] = best_thresh
        logger.info(f"Optimal threshold for '{target_name}': {best_thresh:.3f} (F1: {max_f1:.3f})")

    # Map optimal thresholds to verdict boundaries
    opt_auth = optimal_thresholds.get("authenticity", 0.75)
    opt_fraud = optimal_thresholds.get("fraud", 0.60)
    
    thresholds_config = {
        "ocr_threshold": optimal_thresholds.get("ocr", 0.60),
        "classification_threshold": optimal_thresholds.get("classification", 0.70),
        "extraction_threshold": optimal_thresholds.get("extraction", 0.50),
        "authenticity_genuine": opt_auth,
        "authenticity_suspicious": opt_auth * 0.53, # Scales 0.75 to ~0.40
        "fraud_suspicious": opt_fraud,
        "fraud_genuine": opt_fraud * 0.42 # Scales 0.60 to ~0.25
    }

    # Extract ensemble weights from classification model feature importances
    class_clf = trained_models["classification"]
    feat_importances = np.mean([
        clf.estimator.feature_importances_ for clf in class_clf.calibrated_classifiers_
    ], axis=0)

    kw_imp = float(sum(feat_importances[FEATURE_NAMES.index(f)] for f in FEATURE_NAMES if f.startswith("kw_score_")))
    rg_imp = float(sum(feat_importances[FEATURE_NAMES.index(f)] for f in FEATURE_NAMES if f.startswith("rg_score_")))
    ly_imp = float(sum(feat_importances[FEATURE_NAMES.index(f)] for f in FEATURE_NAMES if f.startswith("ly_score_")))

    total_imp = kw_imp + rg_imp + ly_imp
    if total_imp > 0:
        ensemble_weights = {
            "keyword": kw_imp / total_imp,
            "regex": rg_imp / total_imp,
            "layout": ly_imp / total_imp
        }
    else:
        ensemble_weights = {"keyword": 0.40, "regex": 0.40, "layout": 0.20}

    logger.info(f"Extracted ensemble weights: {ensemble_weights}")

    # Save to local files
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(app_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    scoring_model_path = os.path.join(models_dir, "scoring_model.pkl")
    calibration_model_path = os.path.join(models_dir, "calibration_model.pkl")
    
    with open(scoring_model_path, "wb") as f:
        pickle.dump(trained_models, f)
    logger.info(f"Saved scoring models to: {scoring_model_path}")
    
    calibration_payload = {
        "calibration_info": calibration_info,
        "optimal_thresholds": thresholds_config,
        "ensemble_weights": ensemble_weights
    }

    with open(calibration_model_path, "wb") as f:
        pickle.dump(calibration_payload, f)
    logger.info(f"Saved calibration info metadata to: {calibration_model_path}")

if __name__ == "__main__":
    train_and_save_models()
