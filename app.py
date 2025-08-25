"""
AI Job Copilot — Match %, Missing Skills, Interview Q&A + Suggested Edits
(no embeddings, google-genai tailoring for missing skills)

Setup
-----
1) Secrets / env:
   - Streamlit LLM calls (JD parsing, Q&A) use st.secrets["GOOGLE_API_KEY"] (Gemini 1.5 Flash).
   - Tailoring suggestions use google-genai client and will read key from:
       os.environ["GEMINI_API_KEY"]  (preferred)
     If not set, it will try st.secrets["GOOGLE_API_KEY"].

2) requirements.txt:
   streamlit
   google-generativeai
   google-genai
   PyPDF2
   python-docx

Run
---
   streamlit run app.py
"""

import os
import io
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import docx

# Old SDK for JD parsing + Q&A
import google.generativeai as genai_old

# New SDK for tailoring suggestions based on missing skills
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    HAVE_NEW_GENAI = True
except Exception:
    HAVE_NEW_GENAI = False

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="AI Job Copilot — Match & Q&A + Edits", page_icon="🛠️", layout="wide")
st.title("🛠️ AI Job Copilot — Match & Interview Q&A + Suggested Edits")
st.caption("Streamlit + Gemini | Match score • Missing skills • Evidence • Q&A • Suggestions ")

# Configure old SDK (JD parsing + Q&A)
if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
    genai_old.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.warning("Add GOOGLE_API_KEY to your Streamlit secrets to enable Gemini (JD parse & Q&A).")

LLM_MODEL_OLD = "gemini-1.5-flash"     # for JD parsing + Q&A (google.generativeai)
LLM_MODEL_NEW = "gemini-2.0-flash"     # for tailoring suggestions (google-genai)

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class JDInsights:
    skills: List[str]
    tools: List[str]
    responsibilities: List[str]

# ----------------------------
# Helpers: File parsing
# ----------------------------
def read_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception:
        return ""

def read_docx(file) -> str:
    try:
        document = docx.Document(file)
        texts = [p.text for p in document.paragraphs]
        return "\n".join(texts)
    except Exception:
        return ""

def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    if not chunks and text:
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
    return [c.strip() for c in chunks if c.strip()]

# ----------------------------
# JD Analysis (LLM + regex fallback) — old SDK
# ----------------------------
SKILL_REGEX = re.compile(r"\b([A-Za-z][A-Za-z+.#0-9-]{1,30})\b")

def extract_jd_insights(jd_text: str) -> JDInsights:
    system = (
        "You extract structured fields from a job description. Return strict JSON with keys: "
        "skills (array of short skill keywords), tools (array), responsibilities (array of phrases). "
        "Keep skills/tools deduplicated and concise."
    )
    user = f"JD:\n{jd_text}\n\nReturn JSON only."
    try:
        model = genai_old.GenerativeModel(LLM_MODEL_OLD, system_instruction=system)
        resp = model.generate_content(user)
        raw = (resp.text or "").strip()
        # strip code fences if any
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = re.sub(r"^json", "", raw, flags=re.IGNORECASE).strip()
        data = json.loads(raw)
        skills = sorted({s.strip() for s in data.get("skills", []) if s and s.strip()})
        tools = sorted({s.strip() for s in data.get("tools", []) if s and s.strip()})
        responsibilities = [r.strip() for r in data.get("responsibilities", []) if r and r.strip()]
        return JDInsights(skills=skills, tools=tools, responsibilities=responsibilities)
    except Exception:
        # Naive fallback if LLM parse fails
        tokens = [t for t in SKILL_REGEX.findall(jd_text) if len(t) > 1]
        common = {t for t in tokens if (t.isupper() or t[0].isupper())}
        return JDInsights(skills=sorted(common), tools=[], responsibilities=[])

# ----------------------------
# Interview Q&A — robust JSON (old SDK)
# ----------------------------
QA_SYSTEM = (
    "You are preparing interview questions for a specific job description and candidate resume. "
    "You must output ONLY a JSON array (no prose, no markdown). "
    "Each element must have keys: question (string), ideal_answer (string). "
    "Keep answers concise (6-10 sentences)."
)

QA_USER_TEMPLATE = """JD skills: {skills}
JD responsibilities: {resp}
Resume (verbatim): {resume}

Now output JSON array of 8 items:
[
  {"question":"...","ideal_answer":"..."},
  ...
]
Do NOT include any text outside the JSON array.
"""

def _extract_json_array(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = re.sub(r"^json", "", t, flags=re.IGNORECASE).strip()
    try:
        json.loads(t)
        return t
    except Exception:
        pass
    lb = t.find("[")
    rb = t.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        cand = t[lb:rb+1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            return ""
    return ""

def _template_fallback_qa(jd_skills: List[str]) -> List[Dict[str, str]]:
    skills = jd_skills[:8] or ["Problem Solving", "System Design", "Testing", "APIs"]
    qs = []
    tech = skills[:max(1, min(3, len(skills)))]
    behv = ["A challenging project you shipped end-to-end",
            "Handling conflicting priorities or deadlines",
            "Receiving critical feedback and acting on it"]
    design = ["Design a scalable service for high read/write traffic",
              "Choose data storage and caching for low-latency APIs"]
    for s in tech:
        qs.append({
            "question": f"Deep-dive: How have you applied {s} in production? Walk through a concrete example with trade-offs.",
            "ideal_answer": "Describe context, constraints, your role, approach, tools, metrics, and outcomes. Mention bottlenecks, trade-offs, and what you’d do differently."
        })
    for b in behv:
        qs.append({
            "question": f"Behavioral: {b}. Use STAR (Situation, Task, Action, Result).",
            "ideal_answer": "Provide situation, your task, specific actions, and measurable results. Keep it concise and outcome-focused."
        })
    for d in design[:2]:
        qs.append({
            "question": f"System design: {d}. Outline components and decisions.",
            "ideal_answer": "Cover functional/non-functional requirements, APIs, storage model, scaling strategy, failure handling, monitoring, and trade-offs."
        })
    return qs[:8]

def generate_interview_qa(resume_text: str, jd: JDInsights) -> List[Dict[str, str]]:
    try:
        model = genai_old.GenerativeModel(LLM_MODEL_OLD, system_instruction=QA_SYSTEM)
        prompt = QA_USER_TEMPLATE.format(
            skills=", ".join(jd.skills[:20]),
            resp="; ".join(jd.responsibilities[:12]),
            resume=resume_text[:8000],
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        extracted = _extract_json_array(raw)
        if extracted:
            data = json.loads(extracted)
            clean = []
            for item in data:
                q = (item.get("question") or "").strip()
                a = (item.get("ideal_answer") or "").strip()
                if q and a:
                    clean.append({"question": q, "ideal_answer": a})
            if clean:
                return clean[:8]
    except Exception:
        pass
    return _template_fallback_qa(jd.skills)

# ----------------------------
# Matching & Reporting (lexical only, no embeddings)
# ----------------------------
_nonword = re.compile(r"[^a-z0-9+#./-]+")

def normalize(text: str) -> str:
    return _nonword.sub(" ", text.lower()).strip()

def skill_tokens(skill: str) -> List[str]:
    s = normalize(skill)
    return [t for t in s.split() if t]

def score_skill_in_chunk(skill: str, chunk: str) -> float:
    s_norm = normalize(skill)
    c_norm = normalize(chunk)
    if not s_norm or not c_norm:
        return 0.0
    phrase_hit = 1.0 if s_norm in c_norm else 0.0
    s_toks = set(skill_tokens(skill))
    c_toks = set(c_norm.split())
    if not s_toks:
        return phrase_hit
    overlap = len(s_toks & c_toks) / len(s_toks)
    return max(phrase_hit, overlap)

def compute_skill_matches(jd_skills: List[str], resume_chunks: List[str], sim_threshold: float = 0.75):
    matched, missing, evid_map = [], [], {}
    if not jd_skills or not resume_chunks:
        return matched, jd_skills, {}

    for skill in jd_skills:
        scored = [(i, score_skill_in_chunk(skill, ch)) for i, ch in enumerate(resume_chunks)]
        scored.sort(key=lambda x: x[1], reverse=True)
        evid_map[skill] = [(resume_chunks[i], float(s)) for i, s in scored[:3] if s > 0]
        top_score = scored[0][1] if scored else 0.0
        if top_score >= sim_threshold or any(skill.lower() in ch.lower() for ch in resume_chunks):
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing, evid_map

# ----------------------------
# Tailoring suggestions based on missing skills (new SDK)
# ----------------------------
TAILOR_SYSTEM = """You are a resume editing assistant.

Goal:
- Modify the candidate's existing resume text so that it better aligns with a given Job Description.
- Focus only on highlighting or rephrasing relevant experience to address the list of "missing skills".
- Do NOT fabricate new projects, roles, companies, or dates.
- If a missing skill is not present in the resume, suggest where the candidate could add it (e.g., in SKILLS, PROJECTS, or TRAINING), but clearly mark it as a suggestion.
- Preserve existing content and achievements, only reorganize and edit to improve keyword alignment.

Inputs you will receive:
1. ORIGINAL_RESUME (verbatim text of the candidate’s resume)
2. JD_MISSING_SKILLS (a list of skills the resume currently lacks but the JD requires)

Output requirements:
- Return a Markdown-formatted resume.
- Sections to include: SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION.
- Add a "SUGGESTED ADDITIONS" section at the bottom listing how the candidate could incorporate missing skills they don’t already demonstrate.
- Keep the style professional, concise, and ATS-friendly.
"""

def tailor_resume_for_missing_skills(resume_text: str, missing_skills: List[str]) -> str:
    """
    Uses google-genai client with gemini-2.0-flash to produce suggested edits.
    Feeds ORIGINAL_RESUME and JD_MISSING_SKILLS exactly as requested.
    """
    if not HAVE_NEW_GENAI:
        return "_Install `google-genai` to enable suggestions, or skip this section._"

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return "_Set GEMINI_API_KEY (env) or GOOGLE_API_KEY (secrets) to enable suggestions._"

    client = genai_new.Client(api_key=api_key)
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_text(
                    text=f"ORIGINAL_RESUME:\n{resume_text}\n\nJD_MISSING_SKILLS: {missing_skills}"
                ),
            ],
        ),
    ]
    config = genai_types.GenerateContentConfig(
        system_instruction=[genai_types.Part.from_text(text=TAILOR_SYSTEM)]
    )

    out_chunks = []
    try:
        for chunk in client.models.generate_content_stream(
            model=LLM_MODEL_NEW,
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "text", None):
                out_chunks.append(chunk.text)
    except Exception as e:
        return f"_Suggestion generation error: {e}_"

    return "".join(out_chunks).strip() or "_No suggestions generated._"

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Inputs")
    resume_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    jd_text = st.text_area("Paste Job Description (JD)", height=220, placeholder="Paste the JD here…")
    run = st.button("Run")

if run:
    if not resume_file or not jd_text.strip():
        st.error("Upload a resume and paste the JD.")
        st.stop()

    resume_text = read_pdf(resume_file) if resume_file.name.lower().endswith(".pdf") else read_docx(resume_file)
    if not resume_text.strip():
        st.error("Could not read resume content. Try another file.")
        st.stop()

    with st.spinner("Analyzing JD…"):
        jd = extract_jd_insights(jd_text)

    chunks = chunk_text(resume_text, max_chars=350)
    matched, missing, evid_map = compute_skill_matches(jd.skills, chunks, sim_threshold=0.76)
    total = len(jd.skills) if jd.skills else 1
    match_pct = int(round(100 * (len(matched) / total)))

    st.subheader("📊 Job Readiness Report")
    st.metric(label="Skill Match %", value=f"{match_pct}%")
    st.write("✅ **Matched skills**:", ", ".join(matched) if matched else "None")
    st.write("❌ **Missing skills**:", ", ".join(missing) if missing else "None")

    with st.expander("Evidence (top resume lines per JD skill)", expanded=False):
        for sk in jd.skills:
            st.markdown(f"**{sk}**")
            for sent, score in evid_map.get(sk, []):
                st.caption(f"{score:.2f} — {sent}")
            st.divider()

    with st.spinner("Preparing interview Q&A…"):
        qa = generate_interview_qa(resume_text, jd)

    st.subheader("🎤 Mock Interview Q&A")
    if qa:
        for i, item in enumerate(qa, 1):
            with st.expander(f"Q{i}. {item.get('question','')}"):
                st.markdown(item.get("ideal_answer", ""))
    else:
        st.info("No Q&A generated.")

    # --- New: Suggested Resume Edits (based on missing skills) ---
    st.subheader("✍️ Suggested Resume Edits (based on missing skills)")
    if missing:
        with st.spinner("Generating suggestions..."):
            suggestions_md = tailor_resume_for_missing_skills(resume_text, missing)
        st.markdown(suggestions_md)
    else:
        st.caption("No missing skills detected; suggestions are not needed.")
else:
    st.info("Upload a resume and paste a JD, then click **Run**.")
