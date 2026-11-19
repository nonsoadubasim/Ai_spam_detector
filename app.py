import streamlit as st        
import joblib
import pandas as pd 

print("Streamlit imported successfully")
print("Joblib imported successfully")

#load my trained model and vectorizer

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# CUSTOM CSS - FUTURISTIC AI DESIGN

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1a, #111827, #1e1b4b);
    color: white;
}

h1 {
    color: #38BDF8;
    text-align: center;
    font-size: 45px;
    font-weight: bold;
}

h2, h3 {
    color: #A78BFA;
}

.stTextArea textarea {
    background-color: #172033;
    color: white;
    border: 1px solid #38BDF8;
    border-radius: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #9333EA);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 17px;
    font-weight: bold;
}

.stButton > button:hover {
    border: 1px solid #38BDF8;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #172033;
    padding: 20px;
    border: 1px solid #334155;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

st.divider()

#Configure the app
st.set_page_config(
    page_title= "AI Spam Detector",
    page_icon="💌",
    layout="centered"
)

# Initialize Prediction history

if "history" not in st.session_state:
    st.session_state.history = []


# App title
st.markdown("""
<div style="text-align:center; padding: 20px 0;">
    <h1 style="font-size:45px;">
        💌 AI SPAM DETECTOR
    </h1>
    <p style="color:#A78BFA; font-size:18px;">
        Intelligent Message Protection Powered by Machine Learning
    </p>
    <p style="color:#94A3B8; font-size:14px;">
        Detect • Analyze • Protect
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# About the Project

with st.expander("About This Project"):
    
    st.write("""
    ### AI SPAM DETECTOR
    
    This application uses Machine learning
    to identify whether a text message is Spam or Ham (Not Spam).
    
    **Machine Learning Algorithm:**
    Naive bayes
    
    **Text Preprocessing:**
    TF-IDF Vectorization
    
    **Technologies Used:**
    -Python
    -Scikit-learn
    -Pandas 
    -Streamlit
    -Joblib
    
    **Project GoaL:**
    To automatically identify unwanted spam
    message and distinguish them from legitmate messages.
    """         
    )

st.markdown(
    "<p style='text-align:center;'>"
    "Intelligent Message Classification Powered by Machine learning" 
    "<p>",
    unsafe_allow_html=True
)

st.divider()



st.markdown("""
<div style="
    background: linear-gradient(120deg, #172554, #312E81);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #6366F1;
    margin-bottom: 25px;
">
    <h3 style="color:#38BDF8;">🛡️ Your Intelligent Message Security</h3>
    <p style="color:#E2E8F0; font-size:16px;">
        Protect yourself from unwanted messages.
        Enter any SMS message below and let our
        machine learning model analyze it.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Message input
st.markdown("""
<h2 style="color:#38BDF8; text-align:center;">
    🔍 Message Analysis Center
</h2>
<p style="text-align:center; color:#94A3B8;">
    Paste your message below for intelligent spam detection.
</p>
""", unsafe_allow_html=True)
message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your message here..."  
)

st.caption(f"📝 Characters entered: {len(message)}")

st.divider()

# Prediction button
if st.button("🔍 ANALYZE MESSAGE"):

    
    if message.strip():
        
        #Transform message 
        transformed_message = vectorizer.transform([message])
        
        # Make prediction
        prediction = model.predict(transformed_message)[0]
        probability = model.predict_proba(transformed_message)[0]
        confidence = max(probability) * 100
        
        st.write("Raw Model Prediction:", prediction)
        st.write("Model Classes:", model.classes_)

        
        # Save prediction to history 
        st.session_state.history.append({
            "Message": message,
            "Prediction":str(prediction),
            "Confidence": f"{confidence: .2f}%"
        })
        
        st.divider()
        
        st.markdown("""
        <h2 style="color:#A78BFA; text-align:center;">
        📜 Analysis History
        </h2>
        <p style="color:#94A3B8; text-align:center;">
            Review your previously analyzed messages.
        </p>
        """, unsafe_allow_html=True)
        
        # Prediction statistics
        total_message = len(st.session_state.history)
        
        st.metric(
            label="📩 Total Message Analysed",
            value=total_message
        )
        
        # Count Spam and Ham messages
        spam_count = sum(
            1 for item in st.session_state.history
            if str(item["Prediction"]).lower() == "spam"
            
        )
        
        ham_count = total_message - spam_count
        
        # Display statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📩 Total Analysed", total_message)

        with col2:
            st.metric("⚠️ Spam Detected", spam_count)

        with col3:
            st.metric("✅ Safe Messages", ham_count)
        
        if  st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            st.dataframe(history_df, use_container_width=True)
            
            # Download prediction history
            
            csv = history_df .to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📩 Download History",
                data=csv,
                file_name="spam_detection_history.csv",
                mime="text/csv"
            )    
            
            if st.button("🗑 Clear History"):
                st.session_state.history = []
                st.rerun()
            
        else:
            st.info("No prediction history yet. Analyse a message to get started")    
        
        
        # Display Beautiful Prediction Result

        if str(prediction).lower() == "spam" or prediction == 1:

            st.markdown(f"""
                <div style="
                background: linear-gradient(120deg, #450A0A, #7F1D1D);
                padding: 25px;
                border-radius: 15px;
                border: 1px solid #EF4444;
                text-align: center;
            ">
                <h2 style="color:#FCA5A5;">⚠️ SPAM DETECTED!</h2>
                <p style="color:white;">This message may be unwanted or fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background: linear-gradient(120deg, #064E3B, #065F46);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #22C55E;
            text-align: center;
            ">
            <h2 style="color:#86EFAC;">✅ HAM DETECTED!</h2>
            <p style="color:white;">This message is classified as legitimate.</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            st.caption(
                f"The model is {confidence:.2f}% confident in this prediction."
)
        
        
        
            st.divider()

            with st.expander("👨‍💻 Meet the Developer"):

                st.markdown("""
    <div style="
        background: linear-gradient(120deg, #172554, #312E81);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #6366F1;
    ">
        <h2 style="color:#38BDF8;">ADUBASIM NONSO MIRACLE</h2>
        <h4 style="color:#A78BFA;">Aspiring AI/ML Engineer | Data Analyst</h4>
        <p style="color:#E2E8F0;">
            Passionate about building intelligent applications
            using Python, Machine Learning and Data Analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technical Skills")

    st.write("""
    - Python
    - Machine Learning
    - Data Analysis
    - Scikit-learn
    - Streamlit
    - Natural Language Processing
    """)

    st.markdown("### 🌐 Connect With Me")

    st.markdown(
        "[GitHub](https://github.com/nonsoadubasim)"
    )

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/nonsoadubasimtech)"
    )

    st.caption("AI Spam Detector | Machine Learning Project")
    
    # MODEL PERFORMANCE

    st.divider()

    st.markdown("""
        <h2 style="color:#38BDF8; text-align:center;">
        📊 Model Performance
        </h2>
        <p style="color:#94A3B8; text-align:center;">
        Performance measured on the test dataset.
        </p>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Accuracy", "96.91%")

    with col2:
        st.metric("⚠️ Spam Precision", "100%")

    with col3:
        st.metric("📈 Spam Recall", "76%")
    
    # PROFESSIONAL FOOTER

    st.markdown("""
        <hr style="border: 1px solid #334155;">

        <div style="text-align:center; padding:15px;">

        <h4 style="color:#38BDF8;">
        💌 AI SPAM DETECTOR
        </h4>

        <p style="color:#94A3B8;">
        Intelligent Message Protection Powered by Machine Learning
        </p>

        <p style="color:#64748B; font-size:13px;">
        Built with Python | Scikit-learn | Streamlit
        </p>

        <p style="color:#64748B; font-size:12px;">
        © 2026 Adubasim Nonso Miracle. All Rights Reserved.
        </p>

        </div>
        """, unsafe_allow_html=True)
    