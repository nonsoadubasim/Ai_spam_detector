import streamlit as st        
import joblib
import pandas as pd 

print("Streamlit imported successfully")
print("Joblib imported successfully")

#load my trained model and vectorizer

model = joblib.load("spam_detector_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")


# Custom Css
st.markdown("""
<style>
.stApp{
    background-color: #101827;
    color: white;
}

h1 {
    color: #38BDF8;
    text-align: center;
}    

.stButton>button {
    background-color : #0284C7;
    color: white;
    border-radius: 10px;
    height: 45px;
    widght: 100%
    font-weight: bold;
}

.stTextArea textarea {
    background-color: #1E293B;
    color: white;
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
st.title("💌 AI Spam Detector")

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



st.write("Welcome! to my page Kindly input your message to use AI to detect whether a message is Spam or Ham")

# Message input
st.subheader("Message Analysis")

message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your message here..."  
)

st.divider()

# Prediction button
if st.button("✍ Analysis Message"):

    
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
        
        st.subheader("📜 Prediction History")
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("⚠ Spam Detected", spam_count)
        
        with col2: 
            st.metric("✅ Ham Detected", ham_count)    
        
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
        
        #Display result
        if str(prediction).lower() == "spam" or prediction == 1:
            st.error("⚠ SPAM DETECTED!")
            st.metric("Confidence Score", f"{confidence: .2f}%")
            
        else:
            st.success("✅HAM - This message is safe (Not Spam).")
            st.metric("Confidence Score", f"{confidence: .2f}%")
            
    else:
        st.warning("Please enter a message first") 
        
        
        
st.divider()

with st.expander("👩‍💻 About the Developer"):
    st.subheader("Developer: ADUBASIM NONSO MIRACLE")

    st.write("""
    I am an aspiring AI/ML Engineer and Data Analyst passionate about building intelligent applications
    using Machine learning and Python.
    """)

    st.write("""
    **Technical Skills:**
    - Python
    - Machine Learning
    - Data Analysis
    - Scikit-learn
    - Streamlit
    """)

    st.write("Github: https://github.com/nonsoadubasim")
    st.write("LinkedIn: www.linkedin.com/in/nonsoadubasimtech")
    
    st.caption("AI Spam Detector | Machine Learning Project")