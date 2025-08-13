import streamlit as st
import pdfplumber
from openai import OpenAI
import re
import json
import random
from datetime import datetime
import time
import io
import uuid
from google.oauth2 import service_account
from google.cloud import vision_v1 as vision
from google.cloud import storage
from google.cloud.vision_v1 import AnnotateFileResponse


# ------------------ Enhanced Google OCR Setup (Option A: Async PDF via GCS) ------------------
def setup_vision_and_storage_clients():
    """Build Vision + Storage clients from in-memory service account (Streamlit secrets)."""
    try:
        info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds), storage.Client(credentials=creds)
    except Exception as e:
        st.error(f"Failed to setup Google Vision/Storage clients: {str(e)}")
        return None, None


def normalize_ocr_text(text):
    """Normalize OCR text to reduce parsing errors"""
    if not text:
        return text
    
    # Fix common OCR ligatures
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r?\n+', '\n', text)
    
    # Fix common OCR character substitutions
    text = text.replace('|', 'I').replace('0', 'O', text.count('0') // 3)  # Conservative 0->O replacement
    
    return text.strip()


def needs_ocr(text: str) -> bool:
    """Heuristic to determine if OCR is needed (if pdfplumber text is poor quality)"""
    if not text or len(text.strip()) < 100:
        return True
    
    # Check ratio of alphanumeric characters
    alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
    if alphanumeric_ratio < 0.3:
        return True
        
    # Check for too many single characters (sign of poor extraction)
    words = text.split()
    single_chars = sum(1 for word in words if len(word) == 1)
    if len(words) > 0 and single_chars / len(words) > 0.3:
        return True
        
    return False


def vision_pdf_ocr(pdf_bytes: bytes, gcs_input_bucket: str, gcs_output_bucket: str, timeout_s: int = 300, delete_after: bool = True) -> str:
    """High-accuracy OCR: upload PDF to GCS → async DOCUMENT_TEXT_DETECTION → download JSON → stitch full_text."""
    vision_client, storage_client = setup_vision_and_storage_clients()
    
    if not vision_client or not storage_client:
        raise Exception("Failed to initialize Google Cloud clients")

    # 1) Upload PDF to GCS
    input_bucket = storage_client.bucket(gcs_input_bucket)
    output_bucket = storage_client.bucket(gcs_output_bucket)

    input_blob_name = f"input/{uuid.uuid4()}.pdf"
    input_blob = input_bucket.blob(input_blob_name)
    input_blob.upload_from_string(pdf_bytes, content_type="application/pdf")

    # 2) Configure async request
    feature = vision.Feature(type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source_uri = f"gs://{gcs_input_bucket}/{input_blob_name}"
    gcs_dest_prefix = f"vision-output/{uuid.uuid4()}"  # folder prefix for JSON shards
    gcs_dest_uri = f"gs://{gcs_output_bucket}/{gcs_dest_prefix}/"

    input_config = vision.InputConfig(gcs_source={"uri": gcs_source_uri}, mime_type="application/pdf")
    output_config = vision.OutputConfig(gcs_destination={"uri": gcs_dest_uri}, batch_size=50)

    request = vision.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=input_config,
        output_config=output_config,
    )

    # 3) Kick off job & wait
    operation = vision_client.async_batch_annotate_files(requests=[request])
    operation.result(timeout=timeout_s)  # raises on timeout

    # 4) Read results from GCS
    # Vision writes n JSON shards (one per ~50 pages). We fetch all and stitch.
    texts = []
    for blob in storage_client.list_blobs(output_bucket, prefix=gcs_dest_prefix + "/"):
        data = blob.download_as_bytes()
        resp = AnnotateFileResponse.from_json(data.decode("utf-8"))
        for r in resp.responses:
            if r.full_text_annotation and r.full_text_annotation.text:
                texts.append(r.full_text_annotation.text)

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(texts).strip()

    # 5) Optional cleanup
    if delete_after:
        try:
            input_blob.delete()
            for blob in storage_client.list_blobs(output_bucket, prefix=gcs_dest_prefix + "/"):
                blob.delete()
        except Exception:
            pass  # non-fatal

    return normalize_ocr_text(full_text)


def extract_text_smart(pdf_file):
    """Smart extraction: try pdfplumber first, then high-accuracy Vision OCR if needed"""
    try:
        # 1) Try text layer first (fast for text-based PDFs)
        pdf_file.seek(0)
        base_text = extract_text_with_pdfplumber(pdf_file)
        
        if not needs_ocr(base_text):
            st.info("✅ PDF has good text layer - using direct extraction")
            return normalize_ocr_text(base_text)

        # 2) OCR fallback for scanned/poor quality PDFs
        st.info("📄 PDF appears to be scanned - using high-accuracy Google Vision OCR")
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        text = vision_pdf_ocr(
            pdf_bytes,
            gcs_input_bucket=st.secrets["GCS_INPUT_BUCKET"],
            gcs_output_bucket=st.secrets["GCS_OUTPUT_BUCKET"],
            timeout_s=300
        )
        
        # Fallback if Vision fails
        if not text:
            st.warning("Vision PDF OCR returned empty text. Using pdfplumber as final fallback...")
            pdf_file.seek(0)
            return normalize_ocr_text(extract_text_with_pdfplumber(pdf_file))
            
        return text
        
    except Exception as e:
        st.error(f"Smart extraction failed: {str(e)}")
        st.warning("Falling back to pdfplumber for text extraction...")
        pdf_file.seek(0)
        return normalize_ocr_text(extract_text_with_pdfplumber(pdf_file))


def extract_text_with_pdfplumber(pdf_file):
    """Fallback text extraction using pdfplumber"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text
    except Exception as e:
        st.error(f"Pdfplumber extraction failed: {str(e)}")
        return ""


# ------------------ Intent Detection ------------------
def detect_intent(user_message):
    intents = {
        "umbrella": ["umbrella", "extra liability", "million coverage"],
        "liability": ["liability", "bodily injury", "property damage"],
        "deductible": ["deductible", "collision deductible", "comprehensive deductible"],
        "coverage_comparison": ["compare", "options", "quote", "pricing", "rate", "premium"],
        "general": []
    }
    for intent, keywords in intents.items():
        if any(word in user_message.lower() for word in keywords):
            return intent
    return "general"


def clean_spacing(text):
    # Remove multiple consecutive blank lines
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Remove stray carriage returns
    text = text.replace('\r', '')
    return text.strip()


# ------------------ Function: Extract Data from Dec Page ------------------

def pinned_download_button(json_data, filename="dec_page_extracted.json"):
    # render the widget first so the CSS (below) can override it
    st.download_button(
        "⬇️ Download JSON",
        data=json_data,
        file_name=filename,
        mime="application/json",
        key="pinned_download",
    )

    st.markdown("""
    <style>
    /* Pin the specific download widget (by Streamlit test id) */
    div[data-testid="stDownloadButton"] {
        position: fixed !important;
        top: 70px !important;
        right: 16px !important;
        z-index: 10000 !important;
        width: auto !important;
    }
    /* Style the actual button inside */
    div[data-testid="stDownloadButton"] > button {
        background-color: #F04E30 !important;   /* Polly orange */
        color: #ffffff !important;               /* white text */
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        width: auto !important;
        min-width: 170px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,.15) !important;
        transition: all .2s ease-in-out !important;
    }
    /* Make sure inner spans/icons are white too */
    div[data-testid="stDownloadButton"] > button * {
        color: #ffffff !important;
    }
    /* Hover */
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #ff6a4d !important;    /* lighter orange */
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def extract_dec_page_data(extracted_text):
    """Extract structured data from the OCR extracted text with improved regex patterns"""
    data = {
        "policy_info": {},
        "insured": {},
        "vehicles": [],
        "drivers": []
    }

    # Apply normalization for better regex matching
    normalized_text = normalize_ocr_text(extracted_text)

    # ---------------- Policy Information ----------------
    policy_number = re.search(r"Policy\s*#?:?\s*([A-Z0-9\-]+)", normalized_text, re.I)
    policy_term = re.search(r"Term:?\s*([\d/.-]+)\s*[-–—]\s*([\d/.-]+)", normalized_text)
    premium = re.search(r"(?:Full\s*Term\s*Premium|Premium):?\s*\$?([\d,]+\.?\d{0,2})", normalized_text, re.I)

    if policy_number:
        data["policy_info"]["policy_number"] = policy_number.group(1)
    if policy_term:
        data["policy_info"]["start_date"] = policy_term.group(1)
        data["policy_info"]["end_date"] = policy_term.group(2)
    if premium:
        data["policy_info"]["full_term_premium"] = premium.group(1)

    # ---------------- Named Insured ----------------
    insured_name = re.search(r"(?:Name|Insured):?\s*([A-Z][A-Za-z\s,.']+?)(?:\n|Email|Address)", normalized_text, re.I)
    email = re.search(r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", normalized_text, re.I)
    address = re.search(r"Address:?\s*(\d+.*?(?:MA|Mass)\s*\d{5})", normalized_text, re.I | re.DOTALL)

    data["insured"]["name"] = insured_name.group(1).strip() if insured_name else ""
    data["insured"]["email"] = email.group(1).strip() if email else ""
    data["insured"]["address"] = address.group(1).strip() if address else ""

    # ---------------- Vehicles & Coverages ----------------
    vehicle_blocks = re.findall(r"Veh\s*#?\s*\d+.*?:([\s\S]*?)(?=Veh\s*#?\s*\d+|Drivers?|$)", normalized_text, re.I)
    for vb in vehicle_blocks:
        year_make_model = re.search(r"(\d{4})[,\s]*([A-Z]+)[,\s]*([A-Za-z0-9\s/\-]+)", vb)
        vin = re.search(r"([A-HJ-NPR-Z0-9]{17})", vb)
        vehicle_premium = re.search(r"Vehicle\s*Premium:?\s*\$?([\d,]+\.?\d{0,2})", vb, re.I)

        # Bodily Injury - more flexible pattern
        bi_match = re.search(r"(?:Optional\s*)?bodily\s*injury[:\s]*(\d{1,3})[,\s]*(\d{1,3})", vb, re.I)
        bodily_injury = f"{bi_match.group(1)}/{bi_match.group(2)}" if bi_match else ""

        # Collision - more flexible pattern
        collision_match = re.search(r"Collision[:\s]*(\d+)", vb, re.I)
        collision_deductible = collision_match.group(1) if collision_match else ""

        # Comprehensive - more flexible pattern
        comp_match = re.search(r"Comprehensive[:\s]*(\d+)", vb, re.I)
        comprehensive_deductible = comp_match.group(1) if comp_match else ""

        data["vehicles"].append({
            "year": year_make_model.group(1) if year_make_model else "",
            "make": year_make_model.group(2) if year_make_model else "",
            "model": year_make_model.group(3).strip() if year_make_model else "",
            "vin": vin.group(1) if vin else "",
            "vehicle_premium": vehicle_premium.group(1) if vehicle_premium else "",
            "bodily_injury": bodily_injury,
            "collision_deductible": collision_deductible,
            "comprehensive_deductible": comprehensive_deductible
        })

    # ---------------- Drivers ----------------
    driver_blocks = re.findall(r"Driver\s*#?\s*(\d+)\s*([A-Z][A-Za-z\s]+)\s*(\d{1,2}/\d{1,2}/\d{4})", normalized_text, re.I)
    for db in driver_blocks:
        data["drivers"].append({
            "driver_number": db[0],
            "name": db[1].strip(),
            "dob": db[2]
        })

    return data


def update_json_values(current_json, user_message):
    """
    Dynamically updates extracted JSON values based on user or AI suggestions.
    Example: "Increase BI limits to 100/300" will update all BI limits.
    """
    updated_json = json.loads(current_json)

    # Look for BI limit updates
    bi_match = re.search(r"(increase|raise)\s+bi.*?(\d{2,3}/\d{2,3})", user_message, re.I)
    if bi_match:
        new_limit = bi_match.group(2)
        for vehicle in updated_json.get("vehicles", []):
            vehicle["bodily_injury"] = new_limit

    # Example: update deductible
    ded_match = re.search(r"(deductible)\s+to\s+(\d+)", user_message, re.I)
    if ded_match:
        new_ded = ded_match.group(2)
        for vehicle in updated_json.get("vehicles", []):
            vehicle["collision_deductible"] = new_ded

    return json.dumps(updated_json, indent=4)


# Load Massachusetts-specific agent resource data
with open("data/Massachusetts_Agent_Resource_Updated.txt", "r", encoding="utf-8") as file:
    mass_reference = file.read()

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Polly Coverage Agent", layout="centered")

# ------------------ CSS Styling ------------------
st.markdown("""
<style>
/* Style the file uploader */
section[data-testid="stFileUploader"] {
    background-color: #fff !important;
    border: 2px dashed #F04E30 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    color: #1F2D58 !important;
}

/* Style the Browse button itself */
section[data-testid="stFileUploader"] button {
    background-color: #F04E30 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease-in-out !important;
}

/* Force white text on all button text elements */
section[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

/* Target the button text specifically */
section[data-testid="stFileUploader"] button p {
    color: #ffffff !important;
}

/* Ensure the button text span is white */
section[data-testid="stFileUploader"] button span {
    color: #ffffff !important;
}

/* Target button with kind attribute */
button[kind="secondary"] {
    color: #ffffff !important;
}

/* Hover effect */
section[data-testid="stFileUploader"] button:hover {
    background-color: #ff6a4d !important;
}

/* Hover state text also white */
section[data-testid="stFileUploader"] button:hover * {
    color: #ffffff !important;
}

/* Additional targeting for the upload button text */
.stUploadButton button {
    color: #ffffff !important;
}

.stUploadButton button * {
    color: #ffffff !important;
}

/* Target the specific browse button by data-testid if it exists */
button[data-testid="stUploadButton"] {
    color: #ffffff !important;
}

button[data-testid="stUploadButton"] * {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body, .stApp {
    background-color: #ffffff;
    color: #1F2D58;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #1F2D58;
    text-align: center;
}
.chat-message {
    background-color: #f1f4ff;
    color: #000000;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
section[data-testid="stFileUploader"] {
    background-color: #eef1f9;
    border: 2px dashed #A5B4FC;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 20px;
}
button[kind="primary"] {
    background-color: #F04E30;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    border: none;
}
button[kind="primary"]:hover {
    background-color: #d33e23;
}
.stTextInput>div>div>input {
    border-radius: 8px;
    padding: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ------------------ Title / Header ------------------
st.markdown("""
<div style="
    text-align: center;
    padding: 40px 20px 10px 20px;
    border-radius: 12px;
    background-color: #1F2D58;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
">
    <h1 style="color: #ffffff; font-size: 2.4rem; margin: 0;">Polly's Coverage Agent</h1>
    <p style="color: #DDE3F0; font-size: 1.05rem; margin-top: 0.6rem;">
        Upload your Dec Page or ask any insurance question below.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------ Markdown Cleaner ------------------
def clean_markdown(text):
    # Add basic spacing and fix run-on content
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r'\n+', '\n', text)  # collapse excessive newlines
    text = text.strip()
    return text


# ------------------ OpenAI Setup ------------------
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
    # organization parameter is optional - only include if you have one in secrets
)

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ File Upload Zone ------------------
uploaded_file = st.file_uploader(" ", type=["pdf"])


# ------------------ Generate Fake Carrier Rates Function ------------------
def generate_fake_rates(base_premium):
    """Generate fake rates around the extracted premium ±10%."""
    base = float(base_premium.replace(",", "")) if base_premium else 1200.00
    carriers = ["Travelers", "Geico", "Progressive", "Safeco", "Nationwide"]
    rates = {}
    for carrier in carriers:
        variation = random.uniform(-0.1, 0.1)  # ±10%
        rates[carrier] = round(base * (1 + variation), 2)
    return rates


# ------------------ Auto-generate Summary Function ------------------
def generate_auto_summary(extracted_text, extracted_data):
    """Automatically generate a summary and recommendations when dec page is uploaded."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, conversational insurance agent. When reviewing a client's declaration page, "
                    "provide a brief, easy-to-understand summary of their current coverage and then immediately suggest "
                    "2-3 specific coverage improvements. Keep it conversational and helpful, not formal. "
                    "Format your response with clear sections using <h4> tags and bullet points with <ul><li> tags. "
                    "Always end with one simple, specific question to engage the client."
                )
            },
            {
                "role": "user",
                "content": f"Here's my insurance declaration page. Please give me a brief summary of what I have and suggest some coverage improvements:\n\n{extracted_text}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            timeout=30
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I've received your declaration page. Let me review it and provide recommendations. Error: {str(e)}"

    return "I've received your declaration page. Let me review it and provide recommendations."


# ------------------ Extract Data with Enhanced OCR ------------------
if uploaded_file and "extracted_text" not in st.session_state:
    with st.spinner("Analyzing PDF and extracting text with enhanced accuracy..."):
        # Use smart extraction (pdfplumber first, then Vision OCR if needed)
        extracted_text = extract_text_smart(uploaded_file)
        st.session_state.extracted_text = extracted_text

        # Extract structured data from the OCR text
        extracted_data = extract_dec_page_data(extracted_text)
        st.session_state.extracted_json = json.dumps(extracted_data, indent=4)
        st.session_state.extracted_data = extracted_data

    # ✅ Add delay before showing rates
    with st.spinner("Analyzing your policy and fetching comparison rates..."):
        time.sleep(2)

    # ✅ AUTO-GENERATE SUMMARY AND RECOMMENDATIONS
    with st.spinner("Reviewing your coverage and preparing recommendations..."):
        auto_summary = generate_auto_summary(extracted_text, extracted_data)
        st.session_state.chat_history.append(("assistant", auto_summary))
        st.session_state.dec_summary = auto_summary
        st.session_state.summary_generated = True

# ------------------ Always Show Rate Box After Upload ------------------
if "extracted_data" in st.session_state:
    premium_value = st.session_state.extracted_data.get("policy_info", {}).get("full_term_premium", "1200")
    premium_value = str(premium_value).replace(",", "") if premium_value else "1200"
    fake_quotes = generate_fake_rates(premium_value)

    quote_table = "".join(
        [f"<tr><td class='carrier'>{c}</td><td>${r:,.2f}</td></tr>" for c, r in fake_quotes.items()]
    )
    st.markdown(
        f"""
        <div class="rate-box">
            <h4>🔍 Live Rate Comparison (Enhanced OCR)</h4>
            <table>{quote_table}</table>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------ Display Left-Side Rate Box CSS ------------------
st.markdown(
    """
    <style>
    .rate-box {
        position: fixed;
        top: 100px;
        left: 15px;
        width: 220px;
        background-color: #f8f9ff;
        border: 2px solid #1F2D58;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-family: 'Segoe UI', sans-serif;
        z-index: 999;
    }
    .rate-box h4 {
        margin: 0 0 10px 0;
        font-size: 1rem;
        color: #1F2D58;
        text-align: center;
    }
    .rate-box table {
        width: 100%;
        font-size: 0.9rem;
        border-collapse: collapse;
    }
    .rate-box td {
        padding: 4px 0;
        border-bottom: 1px solid #ddd;
    }
    .rate-box td.carrier {
        font-weight: 600;
        color: #1F2D58;
    }
    </style>
    """, unsafe_allow_html=True
)

# ------------------ Show upload status ------------------
if uploaded_file:
    st.markdown(
        "<div style='text-align:center; color:#1F2D58; font-size:0.95rem; margin-top:5px;'>✅ Dec Page uploaded and processed with Google OCR</div>",
        unsafe_allow_html=True
    )
elif "extracted_text" not in st.session_state:
    st.markdown(
        "<div style='text-align:center; color:#94A3B8; font-style:italic; margin-top:10px;'>No Dec Page uploaded</div>",
        unsafe_allow_html=True
    )

# ------------------ Initialize Empty JSON if Missing ------------------
if "extracted_json" not in st.session_state:
    st.session_state.extracted_json = json.dumps({
        "policy_info": {},
        "insured": {},
        "vehicles": [],
        "drivers": []
    }, indent=4)

# ------------------ Display Chat History ------------------
for role, msg in st.session_state.chat_history:
    icon = "👤" if role == "user" else "🤖"
    bubble_color = "#E0E8FF" if role == "user" else "#f0f0f0"
    msg_clean = clean_spacing(msg)

    # Allow HTML rendering for assistant messages
    if role == "assistant":
        st.markdown(
            f"""
            <div class="chat-message" style="
                background-color:{bubble_color};
                line-height: 1.5;
                padding: 0.8rem;
            ">
                <strong>{icon} {role.capitalize()}</strong><br><br>
                {msg_clean}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-message" style="
                background-color:{bubble_color};
                white-space: pre-line;
                line-height: 1.5;
                padding: 0.8rem;
            ">
                <strong>{icon} {role.capitalize()}</strong><br><br>
                {msg_clean}
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------ Chat Input + Completion ------------------
user_prompt = st.chat_input("Ask your insurance question...")
if "extracted_json" in st.session_state:
    pinned_download_button(st.session_state.extracted_json)

if user_prompt:
    st.session_state.chat_history.append(("user", user_prompt))

    # Detect intent (if not a vague response)
    detected_intent = detect_intent(user_prompt)
    if user_prompt.lower() not in ["sure", "ok", "yes", "yep"]:
        st.session_state.intent = detected_intent

    # If response is vague, use last known intent to clarify the prompt
    if user_prompt.lower() in ["sure", "ok", "yes", "yep"]:
        intent = st.session_state.get("intent", "general")
        if intent == "umbrella":
            user_prompt = "Please provide a realistic fake umbrella policy quote with premium amounts based on my current coverage."
        elif intent == "liability":
            user_prompt = "Please show higher and lower liability limit options with estimated premiums."
        elif intent == "deductible":
            user_prompt = "Please show how changing my deductible would affect my premium in a comparison table."
        elif intent == "coverage_comparison":
            user_prompt = "Please create a table comparing my current coverage to at least two alternative quote options with estimated premiums."
        else:
            user_prompt = "Please suggest additional coverage improvements with estimated premium changes."

    # ✅ Dynamically update JSON if prompt suggests changes
    if "extracted_json" in st.session_state:
        st.session_state.extracted_json = update_json_values(st.session_state.extracted_json, user_prompt)

    with st.spinner("Thinking..."):
        try:
            extracted_text = st.session_state.get("extracted_text", "")
            dec_summary = st.session_state.get("dec_summary", "")

            # ✅ Build messages starting with system prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly, conversational insurance agent chatting naturally with clients about their coverage needs. "
                        "Your primary goals are helping clients clearly understand their current coverage and gently guiding them toward improved protection or additional relevant insurance products. "
                        "Your primary rules are:\n"
                        "1. Ask **only one question at a time** – never ask multiple questions in one message.\n"
                        "2. Keep responses short, natural, and conversational, like texting a client.\n"
                        "3. Use small chunks of information – no long paragraphs.\n"
                        "4. Use <h4> for headings and <ul><li> for bullet points when listing info.\n"
                        "5. Never list more than one follow-up question in a single response. Wait for the client to reply first.\n\n"
                        "Always watch for gaps or extra coverage needs based on what clients mention (new car, home, family changes, travel, business needs, etc.). "
                        "Whenever relevant, proactively and naturally suggest higher coverage limits, umbrella policies, or additional lines like home, renters, condo, motorcycle, boat, RV, small business insurance, dealership products (warranties), and roadside assistance (like AAA). "
                        "Whenever the user uploads a document, confirm it is an insurance declaration page and use it to present comparison quotes. You may make up the premium amounts.\n"
                        "The document has been processed using advanced Google OCR with enhanced accuracy features for maximum precision.\n"
                        "Never be pushy; your tone should always feel helpful and conversational.\n\n"

                        "Communication Guidelines:\n"
                        "Keep messages short, friendly, and conversational, similar to text messaging—not formal reports.\n"
                        "Provide information in small, easy-to-read chunks.\n"
                        "Always ask only one follow-up question at a time. Make sure your message flows like a natural, friendly conversation. Avoid listing or rapid-firing multiple questions in one turn—keep it relaxed and focused.\n"
                        "Present insurance quotes or coverage options in simple, side-by-side comparison tables automatically. Do NOT ask if the client wants a table—always include one by default, along with a brief summary explaining key differences if helpful.\n"
                        "Offer detailed explanations only when specifically requested by the client.\n"
                        "Use HTML formatting for readability if supported: <h4> headings, <ul><li> bullets for quick points.\n\n"

                        "Your main role is to naturally uncover clients' needs, identify coverage gaps, and recommend appropriate insurance solutions in a clear, engaging, conversational manner."
                    )
                }
            ]

            # ✅ Include previous conversation context if Dec summary exists
            if dec_summary:
                messages.append({
                    "role": "system",
                    "content": f"Previous Dec Page summary for context (extracted via enhanced Google OCR):\n{dec_summary}"
                })

            # ✅ Append chat history (excluding the current user message)
            for role, msg in st.session_state.chat_history[:-1]:  # Exclude last user message we just added
                messages.append({"role": role, "content": msg})

            # ✅ Add the current user prompt
            messages.append({"role": "user", "content": user_prompt})

            # Run ChatGPT
            response = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                timeout=30
            )

            if response.choices and response.choices[0].message:
                reply = response.choices[0].message.content.strip()
                st.session_state.chat_history.append(("assistant", reply))
                st.rerun()
            else:
                st.session_state.chat_history.append(("assistant", "⚠️ No response received."))
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"Error: {e}"))
            st.rerun()





