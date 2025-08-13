import streamlit as st
import pdfplumber
from openai import OpenAI
import re
import json
import random
from datetime import datetime
import time
import io
from pdf2image import convert_from_bytes
from google.cloud import vision
import os


# ------------------ Google OCR Setup ------------------
def setup_google_vision():
    """Initialize Google Cloud Vision client with credentials from Streamlit secrets"""
    try:
        # Use the Google service account JSON stored in Streamlit secrets
        if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
            # Parse the JSON string from secrets
            credentials_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])

            # Create temporary credentials file
            with open("google_credentials.json", "w") as f:
                json.dump(credentials_info, f)

            # Set environment variable
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

            # Initialize the Vision client
            client = vision.ImageAnnotatorClient()
            return client
        else:
            st.error(
                "Google service account JSON not found in Streamlit secrets. Please add 'GOOGLE_SERVICE_ACCOUNT_JSON' to your secrets.")
            return None

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format in GOOGLE_SERVICE_ACCOUNT_JSON: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to setup Google Vision API: {str(e)}")
        return None


def extract_text_with_google_ocr(pdf_file):
    """Extract text from PDF using Google Cloud Vision OCR"""
    try:
        # Initialize Google Vision client
        vision_client = setup_google_vision()
        if not vision_client:
            raise Exception("Google Vision client not initialized")

        # Convert PDF pages to images
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer for potential future use

        # Convert PDF to images (one image per page)
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt='PNG')

        extracted_texts = []

        # Process each page
        for i, image in enumerate(images):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create Google Vision image object
            vision_image = vision.Image(content=img_byte_arr)

            # Perform text detection
            response = vision_client.text_detection(image=vision_image)
            texts = response.text_annotations

            if texts:
                page_text = texts[0].description
                extracted_texts.append(page_text)

            # Check for errors
            if response.error.message:
                raise Exception(f'Google Vision API error: {response.error.message}')

        # Combine all page texts
        full_text = '\n\n--- PAGE BREAK ---\n\n'.join(extracted_texts)
        return full_text

    except Exception as e:
        st.error(f"Google OCR extraction failed: {str(e)}")
        # Fallback to pdfplumber
        st.warning("Falling back to pdfplumber for text extraction...")
        return extract_text_with_pdfplumber(pdf_file)


def extract_text_with_pdfplumber(pdf_file):
    """Fallback text extraction using pdfplumber"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text
    except Exception as e:
        st.error(f"Pdfplumber extraction also failed: {str(e)}")
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
    """Extract structured data from the OCR extracted text"""
    data = {
        "policy_info": {},
        "insured": {},
        "vehicles": [],
        "drivers": []
    }

    # ---------------- Policy Information ----------------
    policy_number = re.search(r"Policy #:\s*(\S+)", extracted_text)
    policy_term = re.search(r"Term:\s*([\d/.-]+)\s*-\s*([\d/.-]+)", extracted_text)
    premium = re.search(r"Full Term Premium:\s*\$([\d,]+\.\d{2})", extracted_text)

    if policy_number:
        data["policy_info"]["policy_number"] = policy_number.group(1)
    if policy_term:
        data["policy_info"]["start_date"] = policy_term.group(1)
        data["policy_info"]["end_date"] = policy_term.group(2)
    if premium:
        data["policy_info"]["full_term_premium"] = premium.group(1)

    # ---------------- Named Insured ----------------
    insured_name = re.search(r"Name:\s*(.*)", extracted_text)
    email = re.search(r"Email:\s*(\S+@\S+)", extracted_text)
    address = re.search(r"Address:\s*(\d+.*?MA\s*\d+)", extracted_text)

    data["insured"]["name"] = insured_name.group(1).strip() if insured_name else ""
    data["insured"]["email"] = email.group(1).strip() if email else ""
    data["insured"]["address"] = address.group(1).strip() if address else ""

    # ---------------- Vehicles & Coverages ----------------
    vehicle_blocks = re.findall(r"Veh #\d+ Company Veh #\d+: ([\s\S]*?)(?=Veh #\d+|Drivers|$)", extracted_text)
    for vb in vehicle_blocks:
        year_make_model = re.search(r"(\d{4}),\s*([A-Z]+),\s*([A-Za-z0-9\s/]+)", vb)
        vin = re.search(r"([A-HJ-NPR-Z0-9]{17})", vb)
        vehicle_premium = re.search(r"Vehicle Premium:\s*\$([\d,]+\.\d{2})", vb)

        # Bodily Injury
        bi_match = re.search(r"Optional bodily injury\s+(\d{1,3},?\d*)\s+(\d{1,3},?\d*)", vb)
        bodily_injury = f"{bi_match.group(1)}/{bi_match.group(2)}" if bi_match else ""

        # Collision
        collision_match = re.search(r"Collision\s+(\d+)", vb)
        collision_deductible = collision_match.group(1) if collision_match else ""

        # Comprehensive
        comp_match = re.search(r"Comprehensive\s+(\d+)", vb)
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
    driver_blocks = re.findall(r"Driver #\s*(\d+)\s*([A-Z\s]+)\s(\d{2}/\d{2}/\d{4})", extracted_text)
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
            model="gpt-5-chat-latest",
            messages=messages,
            max_tokens=15000,
            timeout=30
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I've received your declaration page. Let me review it and provide recommendations. Error: {str(e)}"

    return "I've received your declaration page. Let me review it and provide recommendations."


# ------------------ Extract Data with Google OCR ------------------
if uploaded_file and "extracted_text" not in st.session_state:
    with st.spinner("Extracting text using Google OCR..."):
        # Use Google OCR for text extraction
        extracted_text = extract_text_with_google_ocr(uploaded_file)
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
            <h4>🔍 Live Rate Comparison (OCR Powered)</h4>
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
                        "The document has been processed using advanced Google OCR for maximum accuracy.\n"
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
                    "content": f"Previous Dec Page summary for context (extracted via Google OCR):\n{dec_summary}"
                })

            # ✅ Append chat history (excluding the current user message)
            for role, msg in st.session_state.chat_history[:-1]:  # Exclude last user message we just added
                messages.append({"role": role, "content": msg})

            # ✅ Add the current user prompt
            messages.append({"role": "user", "content": user_prompt})

            # Run ChatGPT
            response = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=messages,
                max_tokens=15000,
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


