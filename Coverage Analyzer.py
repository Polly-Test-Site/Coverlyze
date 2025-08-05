import streamlit as st
import pdfplumber
from openai import OpenAI
import re
import json
import random
from datetime import datetime

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

# ------------------ Function: Extract Data from Dec Page ------------------

def pinned_download_button(json_data, filename="dec_page_extracted.json"):
    # Inject CSS to fix button position and style
    st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            position: fixed;
            top: 70px; /* just above the chat input */
            right: 20px;
            background-color: #F04E30 !important;
            color: white !important;
            font-weight: bold;
            padding: 10px 16px;
            border-radius: 8px;
            z-index: 9999;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download JSON",
        data=json_data,
        file_name=filename,
        mime="application/json",
        key="fixed_download",
    )
def extract_dec_page_data(pdf_path):
    data = {
        "policy_info": {},
        "insured": {},
        "vehicles": [],
        "drivers": []
    }

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    # ---------------- Policy Information ----------------
    policy_number = re.search(r"Policy #:\s*(\S+)", text)
    policy_term = re.search(r"Term:\s*([\d/.-]+)\s*-\s*([\d/.-]+)", text)
    premium = re.search(r"Full Term Premium:\s*\$([\d,]+\.\d{2})", text)

    if policy_number:
        data["policy_info"]["policy_number"] = policy_number.group(1)
    if policy_term:
        data["policy_info"]["start_date"] = policy_term.group(1)
        data["policy_info"]["end_date"] = policy_term.group(2)
    if premium:
        data["policy_info"]["full_term_premium"] = premium.group(1)

    # ---------------- Named Insured ----------------
    insured_name = re.search(r"Name:\s*(.*)", text)
    email = re.search(r"Email:\s*(\S+@\S+)", text)
    address = re.search(r"Address:\s*(\d+.*?MA\s*\d+)", text)

    data["insured"]["name"] = insured_name.group(1).strip() if insured_name else ""
    data["insured"]["email"] = email.group(1).strip() if email else ""
    data["insured"]["address"] = address.group(1).strip() if address else ""

    # ---------------- Vehicles & Coverages ----------------
    vehicle_blocks = re.findall(r"Veh #\d+ Company Veh #\d+: ([\s\S]*?)(?=Veh #\d+|Drivers|$)", text)
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
    driver_blocks = re.findall(r"Driver #\s*(\d+)\s*([A-Z\s]+)\s(\d{2}/\d{2}/\d{4})", text)
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
    api_key=st.secrets["OPENAI_API_KEY"],
    organization=st.secrets["ORGANIZATION"]
)

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ File Upload Zone ------------------
uploaded_file = st.file_uploader(" ", type=["pdf"])

# Only extract once
if uploaded_file and "extracted_text" not in st.session_state:
    with st.spinner("Extracting Dec Page data..."):
        extracted_data = extract_dec_page_data(uploaded_file)
        st.session_state.extracted_json = json.dumps(extracted_data, indent=4)
        with pdfplumber.open(uploaded_file) as pdf:
            extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        st.session_state.extracted_text = extracted_text


# Hide filename and just show status
if uploaded_file:
    st.markdown(
        "<div style='text-align:center; color:#1F2D58; font-size:0.95rem; margin-top:5px;'>✅ Dec Page uploaded</div>",
        unsafe_allow_html=True
    )
elif "extracted_text" not in st.session_state:
    st.markdown(
        "<div style='text-align:center; color:#94A3B8; font-style:italic; margin-top:10px;'>No Dec Page uploaded</div>",
        unsafe_allow_html=True
    )

# ------------------ Display Extracted JSON ------------------
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
    st.markdown(
        f"""
        <div class="chat-message" style="background-color:{bubble_color}">
            <strong>{icon} {role.capitalize()}</strong><br>
            {clean_markdown(msg)}
        </div>
        """, unsafe_allow_html=True
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
            summary_done = st.session_state.get("summary_generated", False)
            dec_summary = st.session_state.get("dec_summary", "")

            # Build messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly, conversational insurance agent chatting naturally with clients about their coverage needs. "
                        "Your primary goals are helping clients clearly understand their current coverage and gently guiding them toward improved protection or additional relevant insurance products. "
                        "Always watch for gaps or extra coverage needs based on what clients mention (new car, home, family changes, travel, business needs, etc.). "
                        "Whenever relevant, proactively and naturally suggest higher coverage limits, umbrella policies, or additional lines like home, renters, condo, motorcycle, boat, RV, small business insurance, dealership products (warranties), and roadside assistance (like AAA). "
                        "Whenever the user uploads a document, confirm it is an insurance declaration page and use it to present comparison quotes. You may make up the premium amounts.\n"
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

            # Initialize session state variables if they don't exist
            if "summary_generated" not in st.session_state:
                st.session_state.summary_generated = False
            if "dec_summary" not in st.session_state:
                st.session_state.dec_summary = None

            # Determine message content
            if extracted_text and not st.session_state.summary_generated:
                # First-time upload: send raw Dec Page for explanation
                messages.append({
                    "role": "user",
                    "content": f"This is my insurance policy. Can you explain what I have?\n\n{extracted_text}"
                })
                st.session_state.summary_generated = True

            elif st.session_state.dec_summary:
                # Follow-up: send stored summary + new question
                messages.append({
                    "role": "user",
                    "content": (
                        f"You previously reviewed my Dec Page. Here is the summary you gave me:\n\n"
                        f"{st.session_state.dec_summary}\n\n"
                        f"My question is: {user_prompt}"
                    )
                })

            else:
                # No Dec Page context: send question as-is
                messages.append({
                    "role": "user",
                    "content": user_prompt
                })

            # Run ChatGPT
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=30000,
                timeout=30
            )

            if response.choices and response.choices[0].message:
                reply = response.choices[0].message.content.strip()
                st.session_state.chat_history.append(("assistant", reply))

                # Save the first assistant response as the Dec summary
                if not st.session_state.get("dec_summary") and st.session_state.get("summary_generated"):
                    st.session_state.dec_summary = reply

                st.rerun()
            else:
                st.session_state.chat_history.append(("assistant", "⚠️ No response received."))
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"Error: {e}"))


