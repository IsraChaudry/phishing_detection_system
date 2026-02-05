import re
import random
import os
import json
import numpy as np

from quick_preprocessing import preprocess_text

def _rule_score(cleaned_text: str) -> float:
    """Enhanced heuristic scoring for phishing-like signals.

    Produces a score in [0, 0.99].
    """
    score = 0.02

    # keyword signals with weights
    keywords = {
        'urgent': 0.16,
        'verify': 0.14,
        'click': 0.12,
        'winner': 0.14,
        'congratulations': 0.12,
        'update': 0.08,
        'payment': 0.14,
        'account': 0.08,
        'password': 0.12,
        'final': 0.10,
        'immediately': 0.12,
        'suspend': 0.13,
        'compromised': 0.14,
        'confirm': 0.12,
        'claim': 0.12,
        'security alert': 0.16,
    }
    for k, w in keywords.items():
        if k in cleaned_text:
            score += w

    # url presence (we replace urls with token 'url' in preprocessing)
    if 'url' in cleaned_text or 'http' in cleaned_text:
        score += 0.20

    # email addresses
    if 'email' in cleaned_text:
        score += 0.08

    # dollar amounts
    if re.search(r'\$\s*\d+', cleaned_text):
        score += 0.10

    # excessive punctuation / all-caps indicators
    if re.search(r'!{2,}', cleaned_text):
        score += 0.06

    # repeated imperative phrases
    if re.search(r'(click here|verify now|update now|confirm your)', cleaned_text):
        score += 0.14

    # clamp and add small randomness
    return max(0.0, min(0.99, score + random.uniform(-0.02, 0.02)))

def hybrid_predict(text: str, model, tokenizer, device='cpu', temperature: float = None):
    """Hybrid predictor: combines lightweight rule heuristics with ML model.

    Returns: (pred_int, probs_array([legit,phish]), error_or_none, cleaned_text, token_count)
    """
    cleaned = preprocess_text(text)

    if len(cleaned) < 10:
        return None, None, "Text too short for analysis", cleaned, 0

    rule = _rule_score(cleaned)

    # If no model provided, return rule-based probs
    if model is None or tokenizer is None:
        phish = rule
        probs = np.array([1.0 - phish, phish], dtype=float)
        pred = 1 if phish > 0.5 else 0
        return int(pred), probs, None, cleaned, min(len(cleaned.split()), 512)

    # Model inference (duplicate minimal logic to avoid circular imports)
    try:
        import torch
    except Exception as e:
        return None, None, f"Torch import error: {e}", cleaned, min(len(cleaned.split()), 512)

    # load calibrated temperature and trust threshold if not provided
    if temperature is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            calib_path = os.path.join(base_dir, 'phishing_model_deployment', 'calibration.json')
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    c = json.load(f)
                    temperature = float(c.get('best_temperature', 1.0))
                    TRUST_THRESHOLD = float(c.get('best_trust_threshold', 0.85))
            else:
                temperature = 1.0
                TRUST_THRESHOLD = 0.85
        except Exception:
            temperature = 1.0
            TRUST_THRESHOLD = 0.85
    else:
        # if temperature provided, still try to read trust threshold from calibration
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            calib_path = os.path.join(base_dir, 'phishing_model_deployment', 'calibration.json')
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    c = json.load(f)
                    TRUST_THRESHOLD = float(c.get('best_trust_threshold', 0.85))
            else:
                TRUST_THRESHOLD = 0.85
        except Exception:
            TRUST_THRESHOLD = 0.85

    encoding = tokenizer.encode_plus(
        cleaned,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_count = encoding['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply temperature scaling to logits to reduce overconfidence
        logits = outputs.logits / float(temperature)
        probs_t = torch.softmax(logits, dim=1)[0].cpu().numpy()

    model_conf = float(probs_t[1])

    # Decision logic
    # Conservative trust threshold for fully trusting model predictions
    TRUST_THRESHOLD = 0.85

    # If the text is short, slightly reduce effective model confidence
    adjusted_model_conf = model_conf
    if token_count < 40:
        adjusted_model_conf = model_conf * 0.92

    # If model_conf is above TRUST_THRESHOLD we accept model (after adjustment)
    if adjusted_model_conf > TRUST_THRESHOLD:
        pred = 1 if adjusted_model_conf > probs_t[0] else 0
        return int(pred), probs_t, None, cleaned, token_count

    # If model is moderately confident, combine with rule score
    if 0.4 < adjusted_model_conf <= TRUST_THRESHOLD:
        combined = (adjusted_model_conf + rule) / 2.0
        phish = combined
        probs = np.array([1.0 - phish, phish], dtype=float)
        pred = 1 if phish > 0.6 else 0
        return int(pred), probs, None, cleaned, token_count

    # model_conf <= 0.4: fallback to rule-only
    phish = rule
    probs = np.array([1.0 - phish, phish], dtype=float)
    pred = 1 if phish > 0.6 else 0
    return int(pred), probs, None, cleaned, token_count
