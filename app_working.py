import streamlit as st
# transformers imported lazily inside load_model() to avoid import-time torch DLL errors
import re
import json
import os
import random
import streamlit.components.v1 as components
import time

st.set_page_config(page_title="AI Phishing Detector", page_icon="🛡️", layout="wide")

# Initialize session state for splash screen
if "splash_time" not in st.session_state:
    st.session_state.splash_time = time.time()

st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    .splash-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        width: 100vw;
        position: fixed;
        top: 0;
        left: 0;
        background: linear-gradient(135deg, #e0f2f1 0%, #f5f7fa 100%);
        z-index: 9999;
    }
    
    .splash-content {
        text-align: center;
    }
    
    .splash-shield {
        font-size: 120px;
        margin-bottom: 20px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    .splash-title {
        font-size: 32px;
        font-weight: 700;
        color: #37474f;
        margin: 20px 0;
    }
    
    .splash-subtitle {
        font-size: 16px;
        color: #90a4ae;
        margin: 10px 0;
    }
    
    .loading-dots {
        font-size: 20px;
        color: #90a4ae;
        margin-top: 30px;
        letter-spacing: 5px;
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    .result-box {
        padding: 32px 24px; 
        border-radius: 16px; 
        margin: 24px 0; 
        font-size: 24px; 
        font-weight: 700; 
        text-align: center; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        letter-spacing: 0.3px;
    }
    
    .phishing {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
        color: #880e4f;
        border-left: 6px solid #d81b60;
    }
    
    .legitimate {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        color: #004d40;
        border-left: 6px solid #26a69a;
    }
    
    .small-muted { font-size: 12px; color: #90a4ae; }
    
    /* Enhance metric cards */
    [data-testid="metric-container"] { 
        border-radius: 12px;
        padding: 16px;
        background: linear-gradient(135deg, #f5f7fa 0%, #fafbfc 100%);
        border: 1px solid #e3f2fd;
    }
    
    /* Sidebar styling */
    [data-testid="sidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #eceff1;
    }
    
    /* Header styling */
    h1 { 
        color: #37474f;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    h2, h3, h4 {
        color: #455a64;
        font-weight: 600;
    }
    
    /* Button styling */
    button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: none;
    }
    
    /* Input styling */
    textarea, input {
        border-radius: 8px;
        border: 1px solid #cfd8dc;
        font-size: 14px;
        background-color: #f8f9fa;
    }
    
    textarea:focus, input:focus {
        border-color: #90caf9 !important;
        background-color: #f5f5f5;
        box-shadow: 0 0 0 3px rgba(144, 202, 249, 0.1);
    }
    
    /* Expander styling */
    [data-testid="expander"] {
        border-radius: 8px;
        border: 1px solid #eceff1;
        background-color: #fafbfc;
    }
    
    /* Progress bar color */
    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #a5d6a7 0%, #81c784 100%);
    }
    
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        # import torch here to avoid top-level import errors on machines
        # where Torch's DLLs or runtime are not available.
        global torch
        try:
            import torch
        except Exception as e:
            st.error(f"Torch import failed: {e}\nPlease ensure a compatible PyTorch build is installed.")
            st.stop()
        # import transformers classes lazily (they import torch internally)
        global DistilBertTokenizer, DistilBertForSequenceClassification
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        except Exception as e:
            st.error(f"Transformers import failed: {e}\nPlease install 'transformers' and retry.")
            st.stop()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'phishing_model_deployment')
        
        tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        model = DistilBertForSequenceClassification.from_pretrained(os.path.join(model_dir, 'model'))
        
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        model.eval()
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_phishing(text, model, tokenizer, device='cpu'):
    cleaned_text = clean_text(text)
    
    if len(cleaned_text) < 10:
        return None, None, "Text too short for analysis", cleaned_text, 0

    # If model is not provided (demo mode), use a lightweight heuristic
    if model is None:
        keywords = ['urgent','verify','click','winner','congratulations','update','payment','account','password','prize','bank','lottery','final','immediately']
        score = 0.05
        for k in keywords:
            if k in cleaned_text:
                score += 0.12
        # clamp
        score = max(0.0, min(0.99, score + random.uniform(-0.05, 0.05)))
        probs = [1.0 - score, score]
        pred = 1 if probs[1] > probs[0] else 0
        return int(pred), probs, None, cleaned_text, min(len(cleaned_text.split()), 512)
    
    encoding = tokenizer.encode_plus(
        cleaned_text,
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
    # ensure torch is available (defensive)
    if 'torch' not in globals():
        try:
            import torch
            globals()['torch'] = torch
        except Exception as e:
            return None, None, f"Torch import error: {e}", cleaned_text, token_count

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    return pred.item(), probs[0].cpu().numpy(), None, cleaned_text, token_count

def get_action_recommendation(confidence):
    """Return actionable recommendation based on phishing confidence."""
    if confidence > 0.90:
        return "🚨 QUARANTINE IMMEDIATELY", "HIGH RISK", "This email shows very strong phishing indicators. Quarantine immediately and inform the sender's IT department.", "#ff0000"
    elif confidence > 0.80:
        return "⚠️ REQUIRE URGENT REVIEW", "HIGH RISK", "Strong phishing signals detected. Manual review by security team strongly recommended before delivery.", "#ff6600"
    elif confidence > 0.65:
        return "⚡ FLAG FOR REVIEW", "MEDIUM RISK", "Suspicious patterns detected. Recommend security team review before user sees this.", "#ff9900"
    elif confidence > 0.40:
        return "⚠️ FLAG SUSPICIOUS", "MEDIUM RISK", "Some unusual patterns detected. Consider additional verification if requesting sensitive info.", "#ffcc00"
    else:
        return "✓ SAFE TO DELIVER", "LOW RISK", "This email appears legitimate. No major phishing indicators detected.", "#00aa00"

def main():
    # Display splash screen using st.empty() so it can be cleared
    splash_placeholder = st.empty()
    
    splash_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh; width: 100vw; position: fixed; top: 0; left: 0; background: linear-gradient(135deg, #e0f2f1 0%, #f5f7fa 100%); z-index: 9999;">
        <div style="text-align: center;">
            <div style="font-size: 120px; margin-bottom: 20px; animation: pulse 2s ease-in-out infinite;">🛡️</div>
            <div style="font-size: 32px; font-weight: 700; color: #37474f; margin: 20px 0;">Phishing Detection System</div>
            <div style="font-size: 16px; color: #90a4ae; margin: 10px 0;">Powered by DistilBERT</div>
            <div style="font-size: 20px; color: #90a4ae; margin-top: 30px; letter-spacing: 5px;">●●●</div>
        </div>
    </div>
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    </style>
    """
    
    splash_placeholder.markdown(splash_html, unsafe_allow_html=True)
    time.sleep(2.5)  # Show splash for 2.5 seconds
    splash_placeholder.empty()  # Clear the splash
    
    st.title("Phishing Detection System")
    st.markdown("### DistilBERT-Based Email Security Analysis")
    st.divider()
    
    # Sidebar: demo toggle and model/runtime info
    demo_mode = st.sidebar.checkbox("Demo Mode (No Model)", value=True, help="Run UI without loading the ML model")
    device = 'cpu'
    if demo_mode:
        model = None
        tokenizer = None
        config = {'test_metrics': {'accuracy': 0.95, 'precision': 0.90, 'recall': 0.90, 'f1_score': 0.90}}
        with st.sidebar:
            st.write("**Configuration**")
            st.info("Demo mode enabled — using heuristic classification")
    else:
        with st.spinner("Loading model..."):
            model, tokenizer, config = load_model()
        # Device selection (only when model is loaded)
        try:
            import torch as _torch
            available_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        except Exception:
            available_device = 'cpu'
        with st.sidebar:
            st.write("**Configuration**")
            st.write(f"Model Path: phishing_model_deployment")
            st.write(f"Device: {available_device}")
            device = st.selectbox("Select Device:", options=[available_device, 'cpu'] if available_device == 'cuda' else ['cpu'], index=0)

        with st.sidebar:
            st.divider()
            st.subheader("Model Performance")
            if 'test_metrics' in config:
                metrics = config['test_metrics']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                    st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")
    
    st.markdown("---")
    
    input_method = st.radio(
        "Select Input Method:",
        ["Paste Email Text", "Try Sample Emails"],
        horizontal=True
    )
    
    email_text = ""
    
    if input_method == "Paste Email Text":
        email_text = st.text_area(
            "Email Content:",
            height=200,
            placeholder="Paste the email subject and body here...",
            help="Enter any email text for analysis"
        )
    else:
        samples = {
            "Phishing: Lottery Winner": "Congratulations! You've won $1,000,000 in our lottery! Click here immediately to claim your prize now!",
            "Phishing: Account Verification": "URGENT: Your account has been compromised. Verify your identity immediately by clicking this link.",
            "Phishing: Payment Overdue": "FINAL NOTICE: Your payment is overdue. Update your billing information now to avoid termination.",
            "Legitimate: Team Meeting": "Hi team, reminder about tomorrow's meeting at 2 PM in conference room B.",
            "Legitimate: Package Delivery": "Your package has been delivered successfully. Thank you for your order!",
            "Legitimate: Meeting Notes": "Meeting notes from yesterday are now available on the shared drive.",
        }
        
        selected_sample = st.selectbox("Select Sample Email:", list(samples.keys()))
        email_text = samples[selected_sample]
        st.text_area("Email Content:", email_text, height=150, disabled=True)
    
    if st.button("Analyze Email", type="primary", use_container_width=True):
        if not email_text or len(email_text.strip()) < 10:
            st.warning("Please enter email text (minimum 10 characters)")
        else:
            with st.spinner("Analyzing email..."):
                # Use hybrid predictor (rule-based backup + model)
                try:
                    from hybrid_prediction_system import hybrid_predict
                    prediction, probabilities, error, cleaned_text, token_count = hybrid_predict(email_text, model, tokenizer, device=device)
                except Exception as e:
                    # fallback to original predict_phishing if hybrid import fails
                    prediction, probabilities, error, cleaned_text, token_count = predict_phishing(email_text, model, tokenizer, device=device)
                
                if error:
                    st.error(f"Analysis Error: {error}")
                else:
                    st.markdown("---")
                    st.markdown("## 📊 Security Analysis Results")
                    
                    # Adjusted confidence threshold: require phishing label AND confidence > 0.65
                    confidence = float(probabilities[1]) if probabilities is not None else 0.0
                    legit_conf = float(probabilities[0]) if (probabilities is not None and len(probabilities) > 0) else 0.0
                    is_phishing = (prediction == 1 and confidence > 0.65)

                    if 0.4 < confidence < 0.65:
                        st.warning("⚠️ **Uncertain Detection**: Model confidence is moderate. Consider additional verification.", icon="⚠️")

                    # Risk level badge
                    action_text, risk_level, reason, color = get_action_recommendation(confidence)
                    
                    # Color-coded result box
                    if confidence > 0.80:
                        result_class = "phishing"
                        result_text = "🚨 PHISHING DETECTED"
                    elif confidence > 0.65:
                        result_class = "phishing"
                        result_text = "⚠️ SUSPICIOUS EMAIL"
                    else:
                        result_class = "legitimate"
                        result_text = "✅ LEGITIMATE EMAIL"

                    st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)
                    
                    # Confidence metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚨 Phishing Risk", f"{confidence*100:.1f}%")
                    with col2:
                        st.metric("✅ Legitimacy", f"{legit_conf*100:.1f}%")
                    with col3:
                        st.metric("📍 Risk Level", risk_level)

                    # Confidence bars
                    st.write("**Confidence Distribution:**")
                    col_leg, col_phish = st.columns(2)
                    with col_leg:
                        st.write(f"Legitimate: {legit_conf*100:.1f}%")
                        st.progress(legit_conf)
                    with col_phish:
                        st.write(f"Phishing: {confidence*100:.1f}%")
                        st.progress(confidence)

                    # Clear recommendation section
                    st.markdown("---")
                    st.markdown("## 🎯 Security Recommendation")
                    st.markdown(f'<div style="background-color: {color}; padding: 16px; border-radius: 8px; color: white; font-weight: bold; font-size: 16px;">{action_text}</div>', unsafe_allow_html=True)
                    st.info(f"**Why:** {reason}")

                    # Details expander
                    with st.expander("📋 View Technical Details"):
                        col_orig, col_clean = st.columns(2)
                        with col_orig:
                            st.write("**Original Email Text:**")
                            st.text_area("Original", value=email_text, height=120, disabled=True, label_visibility="collapsed")
                        with col_clean:
                            st.write("**Processed Text:**")
                            st.text_area("Cleaned", value=cleaned_text, height=120, disabled=True, label_visibility="collapsed")
                        
                        st.write("**Analysis Details:**")
                        st.json({
                            'Tokens': token_count,
                            'Phishing Confidence': f"{confidence:.4f}",
                            'Legitimate Confidence': f"{legit_conf:.4f}",
                            'Risk Level': risk_level,
                        })
                        
                        suspicious = [w for w in ['urgent','verify','click','winner','congratulations','update','payment','account','password', 'final', 'immediately'] if w in cleaned_text]
                        if suspicious:
                            st.write(f"**⚠️ Suspicious Keywords Found:** {', '.join(suspicious)}")
                        else:
                            st.write("**✅ No suspicious keywords detected**")
                    
                    st.markdown("---")
                    
                    # Action items based on risk
                    st.markdown("## ✅ Recommended Actions")
                    if confidence > 0.80:
                        st.error("**🛑 HIGH RISK - DO NOT INTERACT**\n\n1. ❌ Do NOT click any links in this email\n2. 🚨 Quarantine immediately\n3. 📞 Report to IT security\n4. ✔️ Verify sender through alternative channel", icon="🛑")
                    elif confidence > 0.65:
                        st.warning("**⚠️ MEDIUM RISK - CAUTION REQUIRED**\n\n1. ⚠️ Do NOT click suspicious links\n2. 🔍 Verify sender identity\n3. 🚫 Do NOT share personal/financial info\n4. 📢 Report if suspicious", icon="⚠️")
                    else:
                        st.success("**✅ LOW RISK - SAFE**\n\n1. ✓ Email appears legitimate\n2. ✓ Standard handling OK\n3. ⚠️ Still exercise caution with requests\n4. 📢 Report unusual behavior", icon="✅")

                    # Details expander
                    with st.expander("View Analysis Details"):
                        st.write("**Cleaned Text:**")
                        st.write(cleaned_text)
                        st.write(f"**Token Count:** {token_count}")
                        suspicious = [w for w in ['urgent','verify','click','winner','congratulations','update','payment','account','password'] if w in cleaned_text]
                        st.write(f"**Suspicious Keywords:** {', '.join(suspicious) if suspicious else 'None'}")
                    

    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #607d8b; padding: 32px 20px; margin-top: 40px;'>
            <p style='font-size: 18px; font-weight: 600;'>Phishing Detection System</p>
            <p style='font-size: 13px; color: #90a4ae; margin-top: 8px;'>Powered by DistilBERT • Built with Streamlit</p>
            <p style='font-size: 11px; color: #b0bec5; margin-top: 16px;'>2025 AI Project</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()