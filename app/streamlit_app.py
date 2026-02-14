import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


# ===========================
# PREDICTOR CLASS (from your notebook)
# ===========================
class DisasterTweetPredictor:
    def __init__(self, model_path="models/bert-distil-best", threshold=0.40):
        self.threshold = threshold
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        text = re.sub(r"http\\S+", "", text).strip()
    
        inputs = self.tokenizer(
        text, return_tensors="pt", padding=True, 
        truncation=True, max_length=128
        ).to(self.device)
    
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.nn.functional.softmax(
                torch.tensor(logits), dim=-1
            ).numpy()
    
        p_disaster = probs[1]
        prediction = 1 if p_disaster >= self.threshold else 0
        confidence = probs[prediction]
    
        disaster_keywords = ["fire", "earthquake", "flood", "accident", "crash", 
                       "explosion", "disaster", "emergency", "destroyed"]
        keywords_found = [kw for kw in disaster_keywords if kw in text.lower()]
    
        explanation = f"Keywords found: {', '.join(keywords_found)}" if keywords_found else "No strong keywords"
    
        return {
        "prediction": prediction,
        "prediction_label": "🚨 DISASTER" if prediction == 1 else "✅ NOT DISASTER",
        "confidence": float(confidence),
        "p_disaster": float(p_disaster),
        "threshold": self.threshold,
        "explanation": explanation,
        "keywords_found": keywords_found,
        "raw_probs": probs.tolist()  # ✅ ADDED THIS LINE - fixes KeyError
    }



# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="🚨 Disaster Tweet Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
.main-header { font-size: 3rem; font-weight: 800; text-align: center; color: #FF4B4B; }
.sub-header { font-size: 1.3rem; text-align: center; color: #555; margin-bottom: 2rem; }
.disaster-box {color: #FF4B4B; background: linear-gradient(135deg, #ffe5e5 0%, #ffcccc 100%); border-left: 6px solid #FF4B4B; padding: 1.5rem; border-radius: 8px; }
.not-disaster-box {color: #00FE00; background: linear-gradient(135deg, #e5ffe5 0%, #ccffcc 100%); border-left: 6px solid #4CAF50; padding: 1.5rem; border-radius: 8px; }
.metric-card { background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%); padding: 1.5rem; border-radius: 12px; text-align: center; }
.tweet-box { background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #4B9FFF; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ===========================
# LOAD MODEL
# ===========================
@st.cache_resource
def load_predictor(threshold=0.40):
    """Load your trained model"""
    try:
        predictor = DisasterTweetPredictor(threshold=threshold)
        st.success("✅ Model loaded successfully!")
        return predictor
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("💡 Make sure models/bert-distil-best/ exists from your training notebook.")
        return None


# ===========================
# HEADER
# ===========================
st.markdown('<div class="main-header">🚨 Disaster Tweet Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Emergency Detection | DistilBERT • 85%+ Accuracy</div>', unsafe_allow_html=True)

st.markdown("---")


# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    threshold = st.slider(
        "Decision Threshold", 0.30, 0.70, 0.40, 0.05,
        help="Lower = more sensitive (catches more disasters, more false alarms)"
    )
    
    predictor = load_predictor(threshold)
    
    st.markdown("---")
    
    st.header("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "85.4%")
        st.metric("Precision", "88.1%")
    with col2:
        st.metric("Recall", "76.1%")
        st.metric("F1-Score", "81.7%")
    
    st.markdown("---")
    
    st.header("🎯 How It Works")
    st.markdown("""
    1. **Input**: Tweet text
    2. **BERT**: Analyzes context & semantics
    3. **Threshold**: p(disaster) > threshold → "Disaster"
    4. **Output**: Prediction + confidence + explanation
    """)
    
    st.markdown("---")
    
    st.header("📈 Threshold Impact")
    st.markdown("""
    | Threshold | Recall | Precision |
    |-----------|--------|-----------|
    | **0.40**  | **0.79** | **0.86** |
    | 0.50      | 0.76    | 0.88     |
    | 0.60      | 0.74    | 0.89     |
    """)
    
    st.markdown("---")
    
    st.header("👨‍💻 This App")
    st.markdown("""
    **Built with:**
    - DistilBERT (66M parameters)
    - PyTorch & Hugging Face
    - Streamlit deployment
    
    **Portfolio Project 2026**
    """)


# ===========================
# MAIN TABS
# ===========================
tab1, tab2, tab3 = st.tabs(["🔍 Single Tweet", "📊 Batch Analysis", "📈 Examples"])


# ===========================
# TAB 1: SINGLE PREDICTION
# ===========================
with tab1:
    st.header("🔍 Single Tweet Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tweet_text = st.text_area(
            "Enter tweet text:",
            height=120,
            placeholder="Forest fire near La Ronge Sask. Canada",
            help="Paste any tweet to see if it describes a real disaster"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    with col2:
        st.info("**Pro tip:** Try disaster reports vs. metaphors like 'my life is a disaster'")
    
    if analyze_btn and predictor and tweet_text:
        with st.spinner("🤖 Analyzing with DistilBERT..."):
            result = predictor.predict(tweet_text)
            
            # Results
            st.markdown("---")
            st.subheader("📊 Results")
            
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if result["prediction"] == 1:
                    st.markdown(
                        f'<div class="disaster-box">'
                        f'<h2>🚨 {result["prediction_label"]}</h2>'
                        f'<p>{result["explanation"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="not-disaster-box">'
                        f'<h2>{result["prediction_label"]}</h2>'
                        f'<p>{result["explanation"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with result_col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            # Probability bar
            # Probability bar - FIXED VERSION
            st.subheader("📊 Probability Breakdown")
            
            p_dis= result['p_disaster']
            p_not = 1 - p_dis
            
            fig = go.Figure(data=[
            go.Bar(
            x=["Not Disaster", "Disaster"],
            y=[p_not*100,p_dis*100],  # ✅ SIMPLIFIED - no raw_probs needed
            text=[f"{p_not*100:.1f}%", f"{p_dis*100:.1f}%"],
            texttemplate="%{text}",
            textposition="auto",
            marker_color=["#4CAF50", "#FF4B4B"],
            width=[0.4, 0.4]
                )
            ])
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis_range=[0, 100],
                yaxis_title="Probability (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Details
            with st.expander("🔍 Detailed Analysis"):
                st.markdown(f"**Input text:** {tweet_text}")
                st.markdown(f"**Threshold:** {result['threshold']:.2f}")
                st.markdown(f"**P(Disaster):** {result['p_disaster']:.3f}")
                st.markdown(f"**Keywords found:** {result['keywords_found']}")
    
    elif analyze_btn and not tweet_text:
        st.warning("⚠️ Enter tweet text first!")


# ===========================
# TAB 2: BATCH ANALYSIS
# ===========================
with tab2:
    st.header("📊 Batch Tweet Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload CSV")
        csv_file = st.file_uploader("📁 CSV with 'text' column", type="csv")
        
        if csv_file and predictor:
            df = pd.read_csv(csv_file)
            
            if "text" not in df.columns:
                st.error("❌ CSV needs a 'text' column!")
            else:
                st.success(f"✅ Loaded {len(df)} tweets")
                st.dataframe(df.head())
                
                if st.button("🚀 Analyze All", type="primary"):
                    progress = st.progress(0)
                    
                    results = []
                    for idx, row in df.iterrows():
                        result = predictor.predict(row["text"])
                        results.append(result)
                        progress.progress((idx + 1) / len(df))
                    
                    df_results = pd.DataFrame(results)
                    df_final = pd.concat([df.reset_index(drop=True), df_results], axis=1)
                    
                    # Summary
                    st.subheader("📈 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(df))
                    with col2:
                        disasters = (df_final["prediction"] == 1).sum()
                        st.metric("Disasters", disasters)
                    with col3:
                        avg_conf = df_final["confidence"].mean() * 100
                        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                    
                    # Pie chart
                    fig = px.pie(
                        df_final,
                        names="prediction_label",
                        title="Disaster vs Non-Disaster",
                        color="prediction_label",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    csv_out = df_final.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv_out,
                        "disaster_analysis_results.csv",
                        "text/csv",
                    )
    
    with col2:
        st.subheader("Quick Multi-Test")
        multi_text = st.text_area(
            "One tweet per line:",
            height=200,
            placeholder="Forest fire near downtown\nMy exam was a disaster\nEarthquake hit",
        )
        
        if st.button("🚀 Test All") and multi_text and predictor:
            tweets = [t.strip() for t in multi_text.split("\n") if t.strip()]
            
            results = [predictor.predict(t) for t in tweets]
            df_quick = pd.DataFrame(results)
            
            st.dataframe(df_quick[["prediction_label", "confidence", "explanation"]])


# ===========================
# TAB 3: EXAMPLES
# ===========================
with tab3:
    st.header("🧪 Test Examples")
    
    examples = [
        ("Forest fire near La Ronge Sask. Canada", "Real disaster report"),
        ("My presentation was a complete disaster", "Metaphorical use"),
        ("Earthquake just hit downtown area", "Real emergency"),
        ("This traffic is killing me", "Hyperbole"),
        ("Building collapsed after explosion", "Major incident"),
        ("I love how my day is going", "Positive tweet"),
    ]
    
    for tweet, desc in examples:
        with st.expander(f"📝 {desc}"):
            result = predictor.predict(tweet)
            st.markdown(f"**Tweet:** {tweet}")
            st.markdown(f"**Result:** {result['prediction_label']} ({result['confidence']*100:.1f}%)")
            st.markdown(f"**Explanation:** {result['explanation']}")


# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <h3>🚨 Disaster Tweet Detection</h3>
    <p>Built with DistilBERT • PyTorch • Streamlit</p>
    <p>Accuracy: 85.4% • Recall: 76.1% • F1: 81.7%</p>
    <p><a href="https://github.com/yourusername/disaster-tweet-detection">GitHub</a> | 
       <a href="https://your-streamlit-app.streamlit.app">Live Demo</a></p>
</div>
""", unsafe_allow_html=True)
