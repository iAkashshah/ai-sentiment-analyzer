import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="😊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .positive { color: #00ff88; font-size: 2.5em; font-weight: bold; }
    .negative { color: #ff4d4d; font-size: 2.5em; font-weight: bold; }
    .neutral { color: #ffd700; font-size: 2.5em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_model()

label_map = {
    "LABEL_0": ("Negative 😠", "#ff4d4d"),
    "LABEL_1": ("Neutral 😐", "#ffd700"),
    "LABEL_2": ("Positive 😊", "#00ff88")
}

st.title("🤖 AI Sentiment Analyzer")
st.caption("Powered by RoBERTa • Instant • Accurate • Free")

tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📊 Batch Upload", "📈 Dashboard"])

with tab1:
    text = st.text_area("Paste your text here", height=150, placeholder="I absolutely love this product! Super fast delivery.")
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text.strip():
            with st.spinner("AI thinking..."):
                res = classifier(text[:512])[0]
                sent_text, color = label_map[res['label']]
                st.markdown(f"<p class='{sent_text.split()[0].lower()}'>{sent_text}</p>", unsafe_allow_html=True)
                st.metric("Confidence", f"{res['score']:.1%}")
                st.progress(res['score'])
                st.caption(f"**Analyzed text:** {text}")
        else:
            st.warning("Please type something")

with tab2:
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded and st.button("🚀 Analyze All Rows", type="primary"):
        with st.spinner("Analyzing all rows..."):
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            texts = df.iloc[:, 0].astype(str).tolist()  # first column
            results = classifier(texts)
            sentiments = [label_map[r['label']][0].split()[0] for r in results]
            scores = [round(r['score'], 4) for r in results]
            df["sentiment"] = sentiments
            df["confidence"] = scores
            st.success(f"✅ Analyzed {len(df)} rows!")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results", csv, f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
            
            st.session_state["df"] = df

with tab3:
    if "df" in st.session_state:
        df = st.session_state["df"]
        counts = df["sentiment"].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, title="Sentiment Distribution", color_discrete_sequence=["#ff4d4d","#ffd700","#00ff88"]), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(x=counts.index, y=counts.values, title="Count by Sentiment", color=counts.index, color_discrete_sequence=["#ff4d4d","#ffd700","#00ff88"]), use_container_width=True)
    else:
        st.info("Upload & analyze a file to see dashboard here")

st.sidebar.success("App deployed successfully! Share the link with anyone.")
