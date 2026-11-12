import streamlit as st
import re
from datetime import datetime
from pypdf import PdfReader
from io import BytesIO

from utils.prediction import predict_offer


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="OfferGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- SESSION STATE ----------------

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


def go_to_analysis():
    st.session_state.navigation = "Offer Analysis"


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FC;
    }

    [data-testid="stSidebar"] {
        background-color: #101B35;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        color: #FFFFFF;
        padding: 12px 0 25px 0;
    }

    .hero {
        background: linear-gradient(120deg, #172B50, #5146A5);
        padding: 38px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        margin-bottom: 12px;
    }

    .hero p {
        color: #E3E8F5;
        font-size: 16px;
    }

    .stat-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #E6EAF2;
        min-height: 120px;
    }

    .stat-title {
        color: #64748B;
        font-size: 14px;
    }

    .stat-value {
        color: #172B50;
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        color: #172B50;
        font-size: 23px;
        font-weight: 750;
        margin: 25px 0 15px 0;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 13px;
        border: 1px solid #E6EAF2;
        min-height: 150px;
    }

    .info-card h4 {
        color: #172B50;
    }

    .info-card p {
        color: #64748B;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        padding: 25px;
    }

    div.stButton > button {
        background-color: #5146A5;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 22px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #3E348C;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- FRAUD INDICATOR ANALYSIS ----------------

def detect_indicators(text):
    """
    Basic rule-based checks for warning signs.
    These checks are supplementary and do not prove fraud.
    """
    indicators = []

    patterns = {
        "Upfront payment demand": (
            r"registration fee|processing fee|security deposit|"
            r"training fee|application fee|verification fee|"
            r"payment required|pay.{0,40}(fee|deposit|charge)"
        ),
        "Unrealistic job promise": (
            r"guaranteed job|guaranteed placement|"
            r"no interview required|no interview needed|"
            r"instant selection|100\s*%\s*placement"
        ),
        "Urgent payment or selection": (
            r"pay immediately|limited seats|urgent payment|"
            r"within 24 hours|immediate payment|offer expires today"
        ),
        "Unverified communication channel": (
            r"whatsapp only|contact through telegram|telegram only"
        )
    }

    for name, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            indicators.append(name)

    # A free email provider is a reason to verify the sender,
    # not evidence by itself that an offer is fraudulent.
    email_pattern = (
        r"\b[\w.+-]+@(gmail\.com|yahoo\.com|outlook\.com|hotmail\.com)\b"
    )
    if re.search(email_pattern, text, re.IGNORECASE):
        indicators.append(
            "Free email provider used - verify the recruiter's identity"
        )

    return indicators


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(uploaded_file):

    pdf = PdfReader(BytesIO(uploaded_file.getvalue()))

    extracted_text = []

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text.append(page_text)

    return "\n".join(extracted_text)


# ---------------- DISPLAY PREDICTION ----------------

def display_prediction(text):

    if not text.strip():
        st.warning("Please provide valid offer information.")
        return

    try:
        with st.spinner("Analyzing offer using BiLSTM model..."):

            result = predict_offer(text)

        prediction = result["prediction"]
        confidence = result["confidence"]
        fake_probability = result["fake_probability"]

        # Rule-based checks are displayed separately from the model score.
        indicators = detect_indicators(text)

        st.markdown("---")
        st.subheader("AI Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", prediction)

        with col2:
            st.metric("Model Confidence", f"{confidence}%")

        with col3:
            st.metric("Fake Probability", f"{fake_probability}%")

        if prediction == "Fake":
            st.error(
                "⚠️ The BiLSTM model classified this content "
                "as potentially fraudulent."
            )
        else:
            st.success(
                "The BiLSTM model classified this content "
                "as potentially legitimate."
            )

        st.markdown("### Suspicious Indicators")

        if indicators:
            for indicator in indicators:
                st.warning(f"⚠️ {indicator}")
        else:
            st.info(
                "No indicators from the basic rule-based checks "
                "were detected. This does not guarantee authenticity."
            )

        st.caption(
            "The model was trained on general job postings, not specifically "
            "formal offer letters. Its confidence score is not a guarantee. "
            "Rule-based indicators are supplementary warning signs. "
            "Verify the employer independently."
        )

        # Save the result and detected indicators in session history.
        st.session_state.analysis_history.insert(
            0,
            {
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Prediction": prediction,
                "Confidence": f"{confidence}%",
                "Fake Probability": f"{fake_probability}%",
                "Suspicious Indicators": (
                    ", ".join(indicators) if indicators else "None detected"
                )
            }
        )

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="brand">🛡️ Intelligent Job and Internship Fraud Detection System</div>',
        unsafe_allow_html=True
    )

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Offer Analysis",
            "Detection History",
            "About Project"
        ],
        label_visibility="collapsed",
        key="navigation"
    )

    st.markdown("---")

    st.markdown("### AI SECURITY")
    st.caption("Protecting students from recruitment fraud.")

    st.markdown("---")
    st.caption("Deep Learning Mini Project")


# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to Intelligent Job and Internship Fraud Detection System 🛡️</h1>
        <p>
            Your intelligent platform for identifying fraudulent
            internship and job opportunities.
        </p>
        <p>
            Analyze offer letters and job postings before trusting
            an employment opportunity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Detection Technology</div>
            <div class="stat-value">BiLSTM</div>
            <p style="color:#64748B;font-size:12px;">
                Deep Learning Model
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Input Formats</div>
            <div class="stat-value">PDF + Text</div>
            <p style="color:#64748B;font-size:12px;">
                Multiple input options
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Detection Categories</div>
            <div class="stat-value">2 Classes</div>
            <p style="color:#64748B;font-size:12px;">
                Real and Fake
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">What Does OfferGuard AI Do?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "OfferGuard AI analyzes internship and job offer information "
        "to identify suspicious recruitment patterns. It uses Natural "
        "Language Processing and Deep Learning to classify opportunities."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📄 Offer Letter Analysis</h4>
            <p>
                Upload a PDF offer letter or paste job-related text
                for automated analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 AI-Based Detection</h4>
            <p>
                Analyze suspicious recruitment language and predict
                whether an opportunity is potentially real or fake.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Common Recruitment Fraud Indicators</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.warning(
            "💰 **Upfront Payment Demands**\n\n"
            "Requests for registration, security or training fees."
        )

        st.warning(
            "📧 **Suspicious Email Domains**\n\n"
            "Recruiters using free email domains instead of "
            "verifiable company communication channels."
        )

    with col2:
        st.warning(
            "💸 **Unrealistic Stipends**\n\n"
            "Unusually high salary or stipend promises without "
            "reasonable job requirements."
        )

        st.warning(
            "📋 **Vague Job Descriptions**\n\n"
            "Missing responsibilities, eligibility criteria "
            "or clear company information."
        )

    st.markdown("---")

    st.markdown("### Ready to verify an opportunity?")

    st.button(
        "Start Offer Analysis →",
        on_click=go_to_analysis
    )


# ---------------- OFFER ANALYSIS PAGE ----------------

elif page == "Offer Analysis":

    st.title("📄 Offer Analysis")

    st.write(
        "Upload an internship/job offer PDF or paste its text "
        "to analyze it using the trained BiLSTM model."
    )

    st.info(
        "The model was trained on job posting descriptions. "
        "Predictions on formal offer letters should be treated "
        "as experimental until separately validated."
    )

    tab1, tab2 = st.tabs([
        "📄 Upload PDF",
        "✍️ Enter Text"
    ])

    with tab1:

        st.subheader("Upload Offer Letter")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"]
        )

        if uploaded_file is not None:

            st.success(f"Uploaded: {uploaded_file.name}")

            if st.button("Analyze PDF"):

                try:
                    extracted_text = extract_pdf_text(uploaded_file)

                    if not extracted_text.strip():
                        st.warning(
                            "No readable text was found. "
                            "This PDF may be scanned or image-based."
                        )
                    else:
                        with st.expander("View extracted PDF text"):
                            st.text(extracted_text)

                        display_prediction(extracted_text)

                except Exception as e:
                    st.error(f"Unable to read PDF: {str(e)}")

    with tab2:

        st.subheader("Enter Offer Details")

        offer_text = st.text_area(
            "Paste job or internship offer information",
            height=250,
            placeholder=(
                "Enter the job title, company details, "
                "job description, stipend, payment conditions..."
            )
        )

        if st.button("Analyze Offer Text"):

            if len(offer_text.strip()) < 20:
                st.warning(
                    "Please enter more offer information "
                    "before starting the analysis."
                )
            else:
                display_prediction(offer_text)


# ---------------- DETECTION HISTORY PAGE ----------------

elif page == "Detection History":

    st.title("📊 Detection History")

    history = st.session_state.analysis_history

    if history:

        st.write("Previously analyzed offers in this session.")

        st.dataframe(
            history,
            width="stretch",
            hide_index=True
        )

        if st.button("Clear Detection History"):
            st.session_state.analysis_history = []
            st.rerun()

    else:
        st.info(
            "No previous analysis records are available in this session."
        )


# ---------------- ABOUT PROJECT PAGE ----------------

elif page == "About Project":

    st.title("ℹ️ About OfferGuard AI")

    st.write("""
    **Project:** AI-Based Fraudulent Internship and Job Offer Detection

    **Objective:** To identify potentially fraudulent internship
    and employment opportunities using Deep Learning.

    **Model:** Bidirectional Long Short-Term Memory (BiLSTM).

    **Technologies:** Python, Streamlit, TensorFlow, NLP and Pandas.

    **Dataset:** Employment Scam Aegean Dataset (EMSCAD).

    **Predictions:** Real (Legitimate) or Fake (Fraudulent).
    """)

    st.markdown("### How the system works")

    st.write("""
    1. User uploads a PDF or enters job-related text.
    2. Text is extracted and cleaned.
    3. The saved tokenizer converts words into numerical sequences.
    4. The trained BiLSTM model processes the sequences.
    5. The system displays a classification and confidence score.
    """)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    OfferGuard AI | Deep Learning Mini Project
    <br>
    Verify Before You Trust.
</div>
""", unsafe_allow_html=True)
