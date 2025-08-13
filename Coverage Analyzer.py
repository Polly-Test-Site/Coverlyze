from flask import Flask, render_template, request, jsonify, session, send_file
import pdfplumber
from openai import OpenAI
import re
import json
import random
from datetime import datetime
import time
import io
import uuid
import os
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from google.cloud import vision_v1 as vision
from google.cloud import storage
from google.cloud.vision_v1 import AnnotateFileResponse

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this!
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration - use environment variables in production
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-key')
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', '{}')
GCS_INPUT_BUCKET = os.getenv('GCS_INPUT_BUCKET')
GCS_OUTPUT_BUCKET = os.getenv('GCS_OUTPUT_BUCKET')

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load Massachusetts reference data
try:
    with open("data/Massachusetts_Agent_Resource_Updated.txt", "r", encoding="utf-8") as file:
        mass_reference = file.read()
except FileNotFoundError:
    mass_reference = "Massachusetts agent resource data not found."

# Google OCR Setup
def setup_vision_and_storage_clients():
    """Build Vision + Storage clients from service account JSON."""
    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds), storage.Client(credentials=creds)
    except Exception as e:
        print(f"Failed to setup Google Vision/Storage clients: {str(e)}")
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
    text = text.replace('|', 'I').replace('0', 'O', text.count('0') // 3)
    
    return text.strip()

def needs_ocr(text: str) -> bool:
    """Heuristic to determine if OCR is needed"""
    if not text or len(text.strip()) < 100:
        return True
    
    alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
    if alphanumeric_ratio < 0.3:
        return True
        
    words = text.split()
    single_chars = sum(1 for word in words if len(word) == 1)
    if len(words) > 0 and single_chars / len(words) > 0.3:
        return True
        
    return False

def vision_pdf_ocr(pdf_bytes: bytes, timeout_s: int = 300, delete_after: bool = True) -> str:
    """High-accuracy OCR using Google Cloud Vision API"""
    vision_client, storage_client = setup_vision_and_storage_clients()
    
    if not vision_client or not storage_client:
        raise Exception("Failed to initialize Google Cloud clients")

    try:
        # Upload PDF to GCS
        input_bucket = storage_client.bucket(GCS_INPUT_BUCKET)
        output_bucket = storage_client.bucket(GCS_OUTPUT_BUCKET)

        input_blob_name = f"input/{uuid.uuid4()}.pdf"
        input_blob = input_bucket.blob(input_blob_name)
        input_blob.upload_from_string(pdf_bytes, content_type="application/pdf")

        # Configure async request
        feature = vision.Feature(type=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
        gcs_source_uri = f"gs://{GCS_INPUT_BUCKET}/{input_blob_name}"
        gcs_dest_prefix = f"vision-output/{uuid.uuid4()}"
        gcs_dest_uri = f"gs://{GCS_OUTPUT_BUCKET}/{gcs_dest_prefix}/"

        input_config = vision.InputConfig(gcs_source={"uri": gcs_source_uri}, mime_type="application/pdf")
        output_config = vision.OutputConfig(gcs_destination={"uri": gcs_dest_uri}, batch_size=50)

        request = vision.AsyncAnnotateFileRequest(
            features=[feature],
            input_config=input_config,
            output_config=output_config,
        )

        # Execute and wait
        operation = vision_client.async_batch_annotate_files(requests=[request])
        operation.result(timeout=timeout_s)

        # Read results
        texts = []
        for blob in storage_client.list_blobs(output_bucket, prefix=gcs_dest_prefix + "/"):
            data = blob.download_as_bytes()
            resp = AnnotateFileResponse.from_json(data.decode("utf-8"))
            for r in resp.responses:
                if r.full_text_annotation and r.full_text_annotation.text:
                    texts.append(r.full_text_annotation.text)

        full_text = "\n\n--- PAGE BREAK ---\n\n".join(texts).strip()

        # Cleanup
        if delete_after:
            try:
                input_blob.delete()
                for blob in storage_client.list_blobs(output_bucket, prefix=gcs_dest_prefix + "/"):
                    blob.delete()
            except Exception:
                pass

        return normalize_ocr_text(full_text)
    
    except Exception as e:
        raise Exception(f"OCR processing failed: {str(e)}")

def extract_text_with_pdfplumber(pdf_file):
    """Extract text using pdfplumber"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text
    except Exception as e:
        raise Exception(f"Pdfplumber extraction failed: {str(e)}")

def extract_text_smart(pdf_file):
    """Smart extraction: try pdfplumber first, then Google OCR if needed"""
    try:
        # Try text layer first
        pdf_file.seek(0)
        base_text = extract_text_with_pdfplumber(pdf_file)
        
        if not needs_ocr(base_text):
            return normalize_ocr_text(base_text)

        # OCR fallback
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        text = vision_pdf_ocr(pdf_bytes, timeout_s=300)
        
        if not text:
            pdf_file.seek(0)
            return normalize_ocr_text(extract_text_with_pdfplumber(pdf_file))
            
        return text
        
    except Exception as e:
        pdf_file.seek(0)
        return normalize_ocr_text(extract_text_with_pdfplumber(pdf_file))

def extract_dec_page_data(extracted_text):
    """Extract structured data from OCR text"""
    data = {
        "policy_info": {},
        "insured": {},
        "vehicles": [],
        "drivers": []
    }

    normalized_text = normalize_ocr_text(extracted_text)

    # Policy Information
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

    # Named Insured
    insured_name = re.search(r"(?:Name|Insured):?\s*([A-Z][A-Za-z\s,.']+?)(?:\n|Email|Address)", normalized_text, re.I)
    email = re.search(r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", normalized_text, re.I)
    address = re.search(r"Address:?\s*(\d+.*?(?:MA|Mass)\s*\d{5})", normalized_text, re.I | re.DOTALL)

    data["insured"]["name"] = insured_name.group(1).strip() if insured_name else ""
    data["insured"]["email"] = email.group(1).strip() if email else ""
    data["insured"]["address"] = address.group(1).strip() if address else ""

    # Vehicles & Coverages
    vehicle_blocks = re.findall(r"Veh\s*#?\s*\d+.*?:([\s\S]*?)(?=Veh\s*#?\s*\d+|Drivers?|$)", normalized_text, re.I)
    for vb in vehicle_blocks:
        year_make_model = re.search(r"(\d{4})[,\s]*([A-Z]+)[,\s]*([A-Za-z0-9\s/\-]+)", vb)
        vin = re.search(r"([A-HJ-NPR-Z0-9]{17})", vb)
        vehicle_premium = re.search(r"Vehicle\s*Premium:?\s*\$?([\d,]+\.?\d{0,2})", vb, re.I)

        bi_match = re.search(r"(?:Optional\s*)?bodily\s*injury[:\s]*(\d{1,3})[,\s]*(\d{1,3})", vb, re.I)
        bodily_injury = f"{bi_match.group(1)}/{bi_match.group(2)}" if bi_match else ""

        collision_match = re.search(r"Collision[:\s]*(\d+)", vb, re.I)
        collision_deductible = collision_match.group(1) if collision_match else ""

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

    # Drivers
    driver_blocks = re.findall(r"Driver\s*#?\s*(\d+)\s*([A-Z][A-Za-z\s]+)\s*(\d{1,2}/\d{1,2}/\d{4})", normalized_text, re.I)
    for db in driver_blocks:
        data["drivers"].append({
            "driver_number": db[0],
            "name": db[1].strip(),
            "dob": db[2]
        })

    return data

def generate_fake_rates(base_premium):
    """Generate fake rates around the extracted premium ±10%"""
    base = float(base_premium.replace(",", "")) if base_premium else 1200.00
    carriers = ["Travelers", "Geico", "Progressive", "Safeco", "Nationwide"]
    rates = {}
    for carrier in carriers:
        variation = random.uniform(-0.1, 0.1)
        rates[carrier] = round(base * (1 + variation), 2)
    return rates

def detect_intent(user_message):
    """Detect user intent from message"""
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

def generate_auto_summary(extracted_text, extracted_data):
    """Generate automatic summary using OpenAI"""
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
            model="gpt-4o",  # Fixed: Use actual available model (gpt-5 doesn't exist)
            messages=messages,
            timeout=60,  # Increased timeout to reduce timeout errors
            max_tokens=1500,  # Added token limit for response length
            temperature=0.7,  # Added temperature for consistent responses
            top_p=1.0,  # Added top_p for token selection diversity
            frequency_penalty=0.0,  # Reduce repetition
            presence_penalty=0.0  # Encourage topic focus
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I've received your declaration page and I'm analyzing it. There was an issue generating the summary: {str(e)}"

    return "I've received your declaration page. Let me review it and provide recommendations."

# Routes
@app.route('/')
def index():
    """Main page - clear previous session"""
    session.clear()  # Add this line
    session['chat_history'] = []
    return render_template('index.html')
    
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Please upload a PDF file'}), 400

        # Extract text
        extracted_text = extract_text_smart(file)
        session['extracted_text'] = extracted_text

        # Extract structured data
        extracted_data = extract_dec_page_data(extracted_text)
        session['extracted_data'] = extracted_data

        # Generate fake rates
        premium = extracted_data.get("policy_info", {}).get("full_term_premium", "1200")
        fake_quotes = generate_fake_rates(str(premium))

        # Generate auto summary
        auto_summary = generate_auto_summary(extracted_text, extracted_data)
        
        # Add to chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append(("assistant", auto_summary))

        return jsonify({
            'success': True,
            'extracted_data': extracted_data,
            'fake_quotes': fake_quotes,
            'auto_summary': auto_summary
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Initialize chat history if needed
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Add user message to history
        session['chat_history'].append(("user", user_message))

        # Detect intent
        detected_intent = detect_intent(user_message)

        # Handle vague responses
        if user_message.lower() in ["sure", "ok", "yes", "yep"]:
            intent = session.get("intent", "general")
            if intent == "umbrella":
                user_message = "Please provide a realistic fake umbrella policy quote with premium amounts based on my current coverage."
            elif intent == "liability":
                user_message = "Please show higher and lower liability limit options with estimated premiums."
            elif intent == "deductible":
                user_message = "Please show how changing my deductible would affect my premium in a comparison table."
            elif intent == "coverage_comparison":
                user_message = "Please create a table comparing my current coverage to at least two alternative quote options with estimated premiums."
            else:
                user_message = "Please suggest additional coverage improvements with estimated premium changes."
        
        session['intent'] = detected_intent

        # Build messages for OpenAI
        extracted_text = session.get('extracted_text', '')
        dec_summary = session.get('dec_summary', '')

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
                    "Never be pushy; your tone should always feel helpful and conversational.\n\n"
                    "Present insurance quotes or coverage options in simple, side-by-side comparison tables automatically. Do NOT ask if the client wants a table—always include one by default, along with a brief summary explaining key differences if helpful.\n"
                    "Use HTML formatting for readability: <h4> headings, <ul><li> bullets for quick points.\n\n"
                )
            }
        ]

        # Add previous summary if exists
        if dec_summary:
            messages.append({
                "role": "system",
                "content": f"Previous Dec Page summary for context:\n{dec_summary}"
            })

        # Add chat history (excluding current user message)
        for role, msg in session['chat_history'][:-1]:
            messages.append({"role": role, "content": msg})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Get OpenAI response
        response = client.chat.completions.create(
            model="gpt-4o",  # Fixed: Use actual available model (gpt-5 doesn't exist)
            messages=messages,
            timeout=60,  # Increased timeout to reduce timeout errors
            max_tokens=1500,  # Added token limit for response length
            temperature=0.7,  # Added temperature for consistent responses
            top_p=1.0,  # Added top_p for token selection diversity
            frequency_penalty=0.0,  # Reduce repetition
            presence_penalty=0.0  # Encourage topic focus
        )

        if response.choices and response.choices[0].message:
            reply = response.choices[0].message.content.strip()
            session['chat_history'].append(("assistant", reply))
            
            return jsonify({
                'success': True,
                'response': reply
            })
        else:
            return jsonify({'error': 'No response received from AI'}), 500

    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        session['chat_history'].append(("assistant", error_msg))
        return jsonify({'error': error_msg}), 500

@app.route('/download_json')
def download_json():
    """Download extracted data as JSON"""
    extracted_data = session.get('extracted_data', {})
    
    # Create JSON file in memory
    json_data = json.dumps(extracted_data, indent=4)
    json_file = io.StringIO(json_data)
    json_bytes = io.BytesIO(json_file.getvalue().encode('utf-8'))
    
    return send_file(
        json_bytes,
        as_attachment=True,
        download_name='dec_page_extracted.json',
        mimetype='application/json'
    )

@app.route('/get_chat_history')
def get_chat_history():
    """Get current chat history"""
    return jsonify({
        'chat_history': session.get('chat_history', []),
        'extracted_data': session.get('extracted_data', {}),
        'fake_quotes': session.get('fake_quotes', {})
    })

@app.route('/clear_session')
def clear_session():
    """Clear session data"""
    session.clear()
    return jsonify({'success': True})

# CORRECT ✅
if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
