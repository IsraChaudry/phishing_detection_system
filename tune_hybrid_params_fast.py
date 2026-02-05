import os
import json
import numpy as np
from math import isclose

from app_working import load_model
from quick_preprocessing import preprocess_text
from hybrid_prediction_system import _rule_score

import torch


def compute_logits_for_val(model, tokenizer, texts):
    """Compute and cache raw logits, token counts, and rule scores once."""
    cache = []
    for i, text in enumerate(texts):
        print(f'  Computing logits {i+1}/{len(texts)}...')
        cleaned = preprocess_text(text)
        enc = tokenizer.encode_plus(cleaned, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        with torch.no_grad():
            out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            logits = out.logits[0].cpu().numpy()
        token_count = enc['input_ids'].shape[1]
        rule = _rule_score(cleaned)
        cache.append({'logits': logits, 'token_count': token_count, 'rule': rule})
    return cache


def hybrid_decision(model_conf, rule_score, trust_threshold, token_count):
    # short-text discount
    adj = model_conf * (0.92 if token_count < 40 else 1.0)
    if adj > trust_threshold:
        return 1, model_conf
    if 0.4 < adj <= trust_threshold:
        combined = (adj + rule_score) / 2.0
        return (1 if combined > 0.6 else 0), combined
    return (1 if rule_score > 0.6 else 0), rule_score


def _accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def _f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if isclose(prec + rec, 0.0):
        return 0.0
    return float(2 * (prec * rec) / (prec + rec))


def main():
    model, tokenizer, _ = load_model()

    val = [
        ("Your account has been suspended. Click here to verify.", 1),
        ("Your account needs verification due to suspicious activity.", 1),
        ("URGENT: Verify your payment information to avoid service interruption.", 1),
        ("Congratulations! You have won a prize. Claim now by clicking the link.", 1),
        ("Reminder: Team lunch tomorrow at 12pm. RSVP if you can make it.", 0),
        ("Hello, your invoice for March is attached. Please review.", 0),
        ("Security alert: Unusual sign-in attempt. Confirm your identity.", 1),
        ("Win $5000 now! This is not a drill, click the secure link.", 1),
        ("FYI: Project docs updated in the shared folder.", 0),
        ("Please confirm your identity to continue using your account.", 1),
        ("Reminder: Submit your timesheet by Friday.", 0),
        ("Action required: Re-validate your payment method to avoid service loss.", 1),
        ("Meeting notes: Attached are the minutes from yesterday's sync.", 0),
        ("Your mailbox storage is almost full. Remove old messages or upgrade now.", 0),
        ("A gift card is waiting for you — claim your reward by visiting this link.", 1),
        ("Final notice: Your subscription payment failed. Update billing immediately.", 1),
        ("Important: Review your recent account activity and secure your account.", 1),
        ("Lunch canceled — will reschedule later this week.", 0),
        ("You have an unpaid invoice. Click the portal to update payment details.", 1),
        ("Hello John, the report draft is attached. Let me know your feedback.", 0),
    ]

    # Precompute logits once
    texts = [t[0] for t in val]
    labels_list = [t[1] for t in val]
    print('Computing logits for validation set...')
    cache = compute_logits_for_val(model, tokenizer, texts)
    print(f'Logits cached for {len(cache)} samples\n')

    temps = [0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0]
    trusts = [0.65, 0.75, 0.80, 0.85, 0.90, 0.95]

    best = None
    grid = []
    
    print(f'Grid searching {len(temps)} temps × {len(trusts)} trusts = {len(temps)*len(trusts)} combinations...')
    count = 0
    for t in temps:
        for tr in trusts:
            preds = []
            probs = []
            labels = []
            for idx, lbl in enumerate(labels_list):
                entry = cache[idx]
                logits = entry['logits']
                token_count = entry['token_count']
                r = entry['rule']
                # Apply temperature to logits and compute softmax
                scaled = logits / float(t)
                exp = np.exp(scaled - np.max(scaled))
                probs_vec = exp / exp.sum()
                p = float(probs_vec[1])
                pred_label, combined_score = hybrid_decision(p, r, tr, token_count)
                preds.append(pred_label)
                probs.append(combined_score)
                labels.append(lbl)
            try:
                probs_arr = np.clip(np.array(probs, dtype=float), 1e-12, 1.0 - 1e-12)
                labels_arr = np.array(labels, dtype=float)
                loss = float(-np.mean(labels_arr * np.log(probs_arr) + (1 - labels_arr) * np.log(1 - probs_arr)))
            except Exception:
                loss = float('inf')
            f1 = _f1(labels, preds)
            acc = _accuracy(labels, preds)
            grid.append({'temp': float(t), 'trust': float(tr), 'f1': float(f1), 'acc': float(acc), 'log_loss': float(loss)})
            if best is None or (f1 > best['f1']) or (f1 == best['f1'] and loss < best['log_loss']):
                best = {'temp': float(t), 'trust': float(tr), 'f1': float(f1), 'acc': float(acc), 'log_loss': float(loss)}
            count += 1
            if count % 10 == 0:
                print(f'  Processed {count}/{len(temps)*len(trusts)} combinations...')

    out = {
        'best_temperature': best['temp'] if best else 1.0,
        'best_trust_threshold': best['trust'] if best else 0.85,
        'best_metrics': best,
        'all_results': grid,
    }

    out_path = os.path.join('phishing_model_deployment', 'calibration.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\nTuning complete!')
    if best:
        print(f'Best temperature: {best["temp"]:.2f}')
        print(f'Best trust threshold: {best["trust"]:.2f}')
        print(f'F1 score: {best["f1"]:.4f}')
        print(f'Accuracy: {best["acc"]:.4f}')
        print(f'Log loss: {best["log_loss"]:.6f}')
    else:
        print('No valid results found during grid search.')
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
