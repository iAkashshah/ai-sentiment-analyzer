import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="😊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .negative { color: #ff4d4d; font-size: 2.8em; font-weight: bold; }
    .neutral { color: #ffd700; font-size: 2.8em; font-weight: bold; }
    .positive { color: #00ff88; font-size: 2.8em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                    device=-1)  # Forces CPU (works perfectly on free Streamlit)

classifier = load_model()

# FIXED label map (model now returns capitalized labels)
label_map = {
    "Negative": ("Negative 😠", "#ff4d4d"),
    "Neutral": ("Neutral 😐", "#ffd700"),
    "Positive": ("Positive 😊", "#00ff88")
}

st.title("🤖 AI Sentiment Analyzer")
st.caption("**Powered by RoBERTa** • Instant • Accurate • Free")

tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📊 Batch Upload", "📈 Dashboard"])

with tab1:
    text = st.text_area("Paste your text here", height=150, 
                        placeholder="I absolutely love this product! Super fast delivery.")
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text.strip():
            with st.spinner("AI thinking..."):
                res = classifier(text[:512])[0]
                label = res['label']                    # e.g. "Neutral"
                score = res['score']
                sent_text, color = label_map[label]
                st.markdown(f"<p class='{label.lower()}'>{sent_text}</p>", unsafe_allow_html=True)
                st.metric("Confidence", f"{score:.1%}")
                st.progress(score)
                st.caption(f"**Text:** {text}")
        else:
            st.warning("Please enter some text.")

with tab2:
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.dataframe(df.head(), use_container_width=True)
        
        text_col = st.selectbox("Select column containing the text", df.columns.tolist())
        
        if st.button("🚀 Analyze All Rows", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(df)} rows..."):
                texts = df[text_col].astype(str).tolist()
                results = classifier(texts)
                
                sentiments = [label_map[r['label']][0].split()[0] for r in results]
                scores = [round(r['score'], 4) for r in results]
                
                df["sentiment"] = sentiments
                df["confidence"] = scores
                
                st.success(f"✅ Analyzed {len(df)} texts!")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", 
                                   csv, 
                                   f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   "text/csv")
                
                st.session_state.df = df

with tab3:
    if "df" in st.session_state:
        df = st.session_state.df
        counts = df["sentiment"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(names=counts.index, values=counts.values, 
                         title="Sentiment Distribution",
                         color_discrete_sequence=["#ff4d4d", "#ffd700", "#00ff88"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(x=counts.index, y=counts.values, 
                          title="Sentiment Counts",
                          color=counts.index,
                          color_discrete_sequence=["#ff4d4d", "#ffd700", "#00ff88"])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Go to Batch Upload tab, analyze a file, and the live dashboard will appear here.")

st.sidebar.success("✅ Fixed & Ready! Share this link with anyone.")

