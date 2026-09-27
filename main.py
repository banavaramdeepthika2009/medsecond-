import os
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from google import genai


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not configured.")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini AI client initialized successfully.")
    except Exception as e:
        print("Gemini initialization error:", str(e))
        client = None


# =========================================================
# GEMINI MODEL
# =========================================================

GEMINI_MODEL = "gemini-3.6-flash"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="MedSecond API",
    description="AI-assisted medical report analysis and chatbot",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# TEMPORARY STORAGE
# =========================================================

last_report_text = ""
last_report_filename = ""


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "MedSecond backend is running successfully!",
        "ai_enabled": client is not None,
        "ai_provider": "Gemini",
        "model": GEMINI_MODEL
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "ai_enabled": client is not None,
        "ai_provider": "Gemini",
        "model": GEMINI_MODEL
    }


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


# =========================================================
# UPLOAD MEDICAL REPORT
# =========================================================

@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):

    global last_report_text
    global last_report_filename

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF medical report."
        )

    temp_file = "uploaded_report.pdf"

    try:

        # -------------------------------------------------
        # SAVE FILE
        # -------------------------------------------------

        contents = await file.read()

        with open(temp_file, "wb") as f:
            f.write(contents)

        # -------------------------------------------------
        # READ PDF
        # -------------------------------------------------

        reader = PdfReader(temp_file)

        extracted_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                extracted_text += page_text + "\n"

        extracted_text = extracted_text.strip()

        # -------------------------------------------------
        # CHECK TEXT
        # -------------------------------------------------

        if not extracted_text:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text from this PDF. "
                    "If this is a scanned/image PDF, OCR is required."
                )
            )

        # -------------------------------------------------
        # STORE REPORT
        # -------------------------------------------------

        last_report_text = extracted_text

        last_report_filename = file.filename

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "success": True,

            "filename": file.filename,

            "message":
                "Medical report uploaded and processed successfully.",

            "pages": len(reader.pages),

            "extracted_text": extracted_text
        }

    except HTTPException:

        raise

    except Exception as e:

        print("UPLOAD ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Could not process the PDF."
        )

    finally:

        if os.path.exists(temp_file):

            os.remove(temp_file)


# =========================================================
# GEMINI AI HELPER
# =========================================================

def ask_gemini(prompt):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text:

            raise Exception("Gemini returned an empty response.")

        return response.text

    except HTTPException:

        raise

    except Exception as e:

        print("=================================================")
        print("GEMINI ERROR")
        print(str(e))
        print("=================================================")

        raise HTTPException(
            status_code=500,
            detail="Gemini AI could not respond. Please try again."
        )


# =========================================================
# AI MEDICAL REPORT ANALYSIS
# =========================================================

@app.post("/analyze-report")
async def analyze_report():

    global last_report_text

    # -----------------------------------------------------
    # CHECK REPORT
    # -----------------------------------------------------

    if not last_report_text:

        raise HTTPException(
            status_code=400,
            detail="Please upload a medical report first."
        )

    # -----------------------------------------------------
    # CHECK AI
    # -----------------------------------------------------

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )

    # -----------------------------------------------------
    # LIMIT REPORT SIZE
    # -----------------------------------------------------

    report_text = last_report_text[:30000]

    # -----------------------------------------------------
    # AI PROMPT
    # -----------------------------------------------------

    prompt = f"""

You are MedSecond AI, a medical information assistant.

Your purpose is to help a patient understand their medical
report in simple, clear language.

IMPORTANT SAFETY RULES:

1. Do NOT diagnose the patient.

2. Do NOT claim to be a doctor.

3. Do NOT replace a qualified healthcare professional.

4. Do NOT recommend starting, stopping, or changing medication.

5. Do NOT invent medical values or information.

6. Clearly distinguish between:
   - What the report actually says
   - Your general explanation of what it may mean

7. Use the reference ranges provided in the report.

8. If a value is outside the report's reference range,
   clearly identify it.

9. If a value is within the provided reference range,
   say that it appears to be within that range.

10. If the report does not provide enough information,
    clearly say so.

11. If something may require urgent medical attention,
    advise the user to seek appropriate medical care.

12. Use simple language suitable for a non-medical user.

Analyze this medical report:

---------------- REPORT START ----------------

{report_text}

---------------- REPORT END ----------------


Return your answer using these sections:

## 1. Report Summary

Give a short and simple overview.

## 2. Key Findings

List the most important findings.

## 3. Values That May Need Attention

For every potentially abnormal value, provide:

- Test name
- Reported value
- Reference range
- Simple explanation

Do not call something abnormal unless the report's
reference range supports that interpretation.

## 4. Possible Meaning in Simple Language

Explain what the important findings may generally mean.

Do not provide a diagnosis.

## 5. Questions to Ask Your Doctor

Give useful questions the patient can discuss with
their doctor.

## 6. Suggested Next Steps

Give safe general suggestions.

Do not prescribe medication.

## 7. Important Medical Disclaimer

Clearly explain that this is AI-generated information
and that medical decisions should be made with a
qualified healthcare professional.

"""


    # -----------------------------------------------------
    # CALL GEMINI
    # -----------------------------------------------------

    answer = ask_gemini(prompt)

    return {

        "success": True,

        "analysis": answer
    }


# =========================================================
# AI CHATBOT
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    global last_report_text

    # -----------------------------------------------------
    # GET USER MESSAGE
    # -----------------------------------------------------

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    # -----------------------------------------------------
    # CHECK AI
    # -----------------------------------------------------

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )

    # -----------------------------------------------------
    # REPORT CONTEXT
    # -----------------------------------------------------

    if last_report_text:

        report_context = f"""

The user has uploaded a medical report.

Use this report as context if the user's question
is related to their report.

Do not invent information that is not contained
in the report.

---------------- UPLOADED REPORT ----------------

{last_report_text[:20000]}

---------------- END REPORT ----------------

"""

    else:

        report_context = """

The user has not uploaded a medical report yet.

"""


    # -----------------------------------------------------
    # MEDSECOND AI PERSONALITY
    # -----------------------------------------------------

    system_prompt = """

You are MedSecond AI Assistant.

You are a friendly, intelligent and supportive AI
assistant inside the MedSecond application.

Your communication should feel natural and conversational.

PERSONALITY:

- Friendly
- Warm
- Helpful
- Clear
- Patient
- Simple
- Supportive

Do not sound like a robotic medical form.

You can help users with:

- Understanding medical terms
- Understanding medical reports
- Explaining laboratory values
- Explaining general health concepts
- Preparing questions for doctors
- Explaining general treatment concepts
- Explaining second opinions
- Explaining how MedSecond works
- Helping users navigate the application

MEDICAL SAFETY RULES:

1. Never claim to be a doctor.

2. Never provide a definitive diagnosis.

3. Never tell the user to start, stop, or change medication.

4. Never present uncertain information as fact.

5. Encourage consultation with a qualified healthcare
   professional when appropriate.

6. If the user describes a possible emergency,
   recommend immediate emergency medical care.

7. Explain medical concepts using simple language.

8. Do not repeatedly give long disclaimers.

9. Answer the user's actual question directly.

10. Do not invent information.

11. If you do not know something, say so.

12. If the question is unrelated to medicine,
    you can answer normally when appropriate.

The user should feel like they are talking to a helpful
AI assistant.

"""


    # -----------------------------------------------------
    # COMPLETE CHAT PROMPT
    # -----------------------------------------------------

    prompt = f"""

{system_prompt}

{report_context}

USER QUESTION:

{message}

Give a clear, natural and useful answer.

If the question is about the uploaded report,
use the report information.

If the question is about MedSecond,
explain the application clearly.

If the question is a general medical question,
explain it in simple language.

Do not invent information.

"""


    # -----------------------------------------------------
    # CALL GEMINI
    # -----------------------------------------------------

    answer = ask_gemini(prompt)

    return {

        "success": True,

        "reply": answer
    }


# =========================================================
# CURRENT REPORT
# =========================================================

@app.get("/current-report")
def current_report():

    if not last_report_text:

        return {

            "uploaded": False,

            "filename": None
        }

    return {

        "uploaded": True,

        "filename": last_report_filename,

        "text": last_report_text
    }


# =========================================================
# CLEAR REPORT
# =========================================================

@app.delete("/clear-report")
def clear_report():

    global last_report_text
    global last_report_filename

    last_report_text = ""

    last_report_filename = ""

    return {

        "success": True,

        "message": "Current report cleared."
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    print("")
    print("==============================================")
    print("       MEDSECOND BACKEND")
    print("==============================================")

    if client:

        print("AI Provider : Gemini")
        print("AI Model    :", GEMINI_MODEL)
        print("AI Status   : ENABLED")

    else:

        print("AI Provider : Gemini")
        print("AI Status   : DISABLED")
        print("Check GEMINI_API_KEY in .env")

    print("Server      : http://127.0.0.1:8000")
    print("==============================================")
    print("")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )