import re
import PyPDF2
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

app = FastAPI()

# Enable CORS (modify if deploying securely)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Client Setup
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
model_name = "moonshotai/kimi-k2-instruct"

# PDF Text Extraction
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Prompt Builder
def build_prompt(resume_text, job_desc):
    return f"""
Compare the following resume and job description and provide the response in this exact format:

Overall ATS Score: <score>/100
Keyword Match: <percentage>%
Skill Match: <percentage>%
Experience & Education Match: <percentage>%
Formatting Quality: <percentage>%

Matched Keywords:
- <keyword1>
- <keyword2>

Missing Keywords:
- <keyword> (type: <type>, years required: <years>, context: <context from JD>)

Improvement Tips:
- <tip1>
- <tip2>

Feedback Report:
<detailed feedback>

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_desc}\"\"\"
"""

# AI API Call
def query_groq_model(prompt):
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert ATS evaluator. Provide the response in the exact format specified in the prompt, with clear section headers and consistent separators (e.g., ':'). Do not deviate from the requested structure."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Regex Parser
def parse_ai_response(response_text):
    def extract_block(label):
        pattern = rf"{label}\s*[:\-]?\s*(.*?)(?=\n\s*(?:###|\w+\s*[:\-]|$))"
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def clean_list(text_block):
        if not text_block:
            return []
        return [item.strip("-• \n").strip() for item in re.split(r"\n|,|\s*-\s*", text_block) if item.strip()]

    # Debug: Print raw AI output
    print("Raw AI Output:\n", response_text)

    # Extract metrics
    ats_score = re.search(r"(?:Overall\s*)?ATS\s*Score\s*[:\-]?\s*(\d{1,3})(?:\s*/\s*\d{1,3})?", response_text, re.IGNORECASE)
    keyword_match = re.search(r"Keyword\s*Match\s*[:\-]?\s*(\d{1,3})\s*(?:%|percent)", response_text, re.IGNORECASE)
    skill_match = re.search(r"Skill\s*Match\s*[:\-]?\s*(\d{1,3})\s*(?:%|percent)", response_text, re.IGNORECASE)
    exp_edu_match = re.search(r"(?:Experience\s*(?:&|and|\/)\s*Education\s*Match)\s*[:\-]?\s*(\d{1,3})\s*(?:%|percent)", response_text, re.IGNORECASE)
    formatting = re.search(r"Formatting\s*Quality\s*[:\-]?\s*(\d{1,3})\s*(?:%|percent)", response_text, re.IGNORECASE)

    # Debug: Print regex matches
    print("ATS Score Match:", ats_score)
    print("Keyword Match:", keyword_match)
    print("Skill Match:", skill_match)
    print("Exp & Edu Match:", exp_edu_match)
    print("Formatting Match:", formatting)

    def safe_int(match, group=1):
        if match and match.group(group):
            value = int(match.group(group))
            return value if 0 <= value <= 100 else 0
        return 0

    # Extract text blocks
    matched_keywords = clean_list(extract_block("Matched Keywords"))
    missing_keywords = clean_list(extract_block("Missing Keywords"))
    improvement_tips = extract_block("Improvement Tips")
    feedback_report = extract_block("Feedback Report")

    return {
        "ATS Score": safe_int(ats_score),
        "Keyword Match": safe_int(keyword_match),
        "Skill Match": safe_int(skill_match),
        "Experience and Education Match": safe_int(exp_edu_match),
        "Formatting Quality": safe_int(formatting),
    }

# Root Endpoint
@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong", "status": "API is live ✅"}, status_code=200)
@app.get("/")
async def read_root():
    return {"message": "ATS API running 🚀..."}

# Analyze Endpoint
@app.post("/analyze")
async def analyze_resume(
    pdf: UploadFile = File(...),
    jobDesc: str = Form(...)
):
    try:
        # Extract resume text
        resume_text = extract_text_from_pdf(pdf.file)
        if "Error" in resume_text:
            return JSONResponse(status_code=400, content={"error": resume_text})

        # Build prompt and query AI
        prompt = build_prompt(resume_text, jobDesc)
        ai_output = query_groq_model(prompt)

        # Parse AI response
        result = parse_ai_response(ai_output)

        # Prepare response
        response = {
            "result": result,
            "ai_output": ai_output
        }

        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
