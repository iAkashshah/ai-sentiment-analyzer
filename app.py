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
    .neutral  { color: #ffd700; font-size: 2.8em; font-weight: bold; }
    .positive { color: #00ff88; font-size: 2.8em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_model()

# ROBUST label handler (works with lowercase, Capitalized, LABEL_0, etc.)
def get_sentiment(raw_label, score):
    lbl = raw_label.strip().lower()
    if "negative" in lbl or lbl == "label_0":
        return "Negative 😠", "#ff4d4d", "negative"
    elif "neutral" in lbl or lbl == "label_1":
        return "Neutral 😐", "#ffd700", "neutral"
    elif "positive" in lbl or lbl == "label_2":
        return "Positive 😊", "#00ff88", "positive"
    else:
        return f"Unknown ({raw_label})", "#ffffff", "neutral"

st.title("🤖 AI Sentiment Analyzer")
st.caption("**Powered by RoBERTa** • Now 100% robust")

tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📊 Batch Upload", "📈 Dashboard"])

with tab1:
    text = st.text_area("Paste your text here", height=150, 
                        placeholder="Roses are red")
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text.strip():
            with st.spinner("AI analyzing..."):
                res = classifier(text[:512])[0]
                raw_label = res['label']
                score = res['score']
                
                # Debug (you can remove this line later)
                st.caption(f"**Debug:** Model returned → **{raw_label}** (score: {score:.1%})")
                
                sent_text, color, css_class = get_sentiment(raw_label, score)
                
                st.markdown(f"<p class='{css_class}'>{sent_text}</p>", unsafe_allow_html=True)
                st.metric("Confidence", f"{score:.1%}")
                st.progress(score)
                st.caption(f"**Text:** {text}")
        else:
            st.warning("Please enter some text.")

with tab2:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded and st.button("🚀 Analyze All Rows", type="primary", use_container_width=True):
        with st.spinner("Analyzing all rows..."):
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            texts = df.iloc[:, 0].astype(str).tolist()   # uses first column
            results = classifier(texts)
            
            sentiments = []
            scores_list = []
            for r in results:
                sent_text, _, _ = get_sentiment(r['label'], r['score'])
                sentiments.append(sent_text.split()[0])
                scores_list.append(round(r['score'], 4))
            
            df["sentiment"] = sentiments
            df["confidence"] = scores_list
            
            st.success(f"✅ Analyzed {len(df)} rows!")
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
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, 
                                   title="Sentiment Distribution",
                                   color_discrete_sequence=["#ff4d4d","#ffd700","#00ff88"]), 
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(x=counts.index, y=counts.values, 
                                   title="Count by Sentiment",
                                   color=counts.index,
                                   color_discrete_sequence=["#ff4d4d","#ffd700","#00ff88"]), 
                            use_container_width=True)
    else:
        st.info("👈 Analyze a file in Batch tab to see dashboard")

st.sidebar.success("✅ Fixed & Robust! Test with 'Roses are red'")
