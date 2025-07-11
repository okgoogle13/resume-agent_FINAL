# document_generator.py
"""
Handles the core logic of generating documents as structured Markdown text.
This version includes improved export quality for DOCX and PDF and minor bug fixes.
"""
from api_clients import GeminiClient
from typing import List, Dict, Any
from docx import Document
from weasyprint import HTML, CSS
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
import json
from config import USER_NAME_PLACEHOLDER

# Initialize the embedding model once when the module is loaded
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_relevant_experiences(question: str, experiences: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Finds the most relevant career experiences based on a query string."""
    if not experiences or not question:
        return []
    
    # Create a single text representation for each experience
    experience_texts = [f"{exp.get('title', '')}. {exp.get('situation', '')} {exp.get('task', '')} {exp.get('action', '')} {exp.get('result', '')}" for exp in experiences]
    
    # Generate embeddings
    question_embedding = embedding_model.encode([question])
    experience_embeddings = embedding_model.encode(experience_texts)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(question_embedding, experience_embeddings)[0]
    
    # Get the indices of the top-k most similar experiences
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [experiences[i] for i in top_indices]

class DocumentGenerator:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def _generate_ai_content(self, prompt: str) -> str:
        """Helper function to call the Gemini API."""
        return self.gemini_client.generate_text(prompt)

    def _create_docx_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a formatted DOCX from Markdown content."""
        doc = Document()
        for line in markdown_content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                p = doc.add_paragraph()
                p.add_run(line[4:]).bold = True
            elif line.startswith('- '):
                # Add paragraph with bullet style, ensuring text is not empty
                p_text = line[2:].strip()
                if p_text:
                    doc.add_paragraph(p_text, style='List Bullet')
            elif line:
                doc.add_paragraph(line)
        
        # Save document to a byte stream
        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)
        return bio.getvalue()

    def _create_pdf_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a styled PDF from Markdown content."""
        html_content = markdown.markdown(markdown_content)
        # Professional styling for the PDF
        css = CSS(string='''
            @page { size: A4; margin: 1.5cm; }
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }
            h1 { font-size: 22pt; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; margin-top: 0;}
            h2 { font-size: 16pt; color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 3px; margin-top: 20px; }
            h3 { font-size: 12pt; color: #34495e; font-weight: bold; margin-top: 15px;}
            ul { list-style-type: disc; padding-left: 20px; }
            p { margin-bottom: 10px; }
            a { color: #2980b9; text-decoration: none; }
        ''')
        return HTML(string=html_content).write_pdf(stylesheets=[css])

    def generate_resume_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]]) -> str:
        """Generates the resume content as a Markdown string."""
        md_parts = [f"# {user_profile.get('full_name', 'Your Name')}"]
        contact_info = " | ".join(filter(None, [user_profile.get('phone'), user_profile.get('email'), user_profile.get('address'), user_profile.get('linkedin_url')]))
        md_parts.append(contact_info)
        
        if user_profile.get('professional_summary'):
            md_parts.append("\n## PROFESSIONAL SUMMARY")
            md_parts.append(f"{user_profile.get('professional_summary')}")
        
        if experiences:
            md_parts.append("\n## PROFESSIONAL EXPERIENCE")
            for exp in experiences:
                md_parts.append(f"\n### {exp.get('title')}")
                md_parts.append(f"**{exp.get('company')}** | *{exp.get('dates')}*")
                bullets = exp.get('resume_bullets', '').split('\n')
                for bullet in bullets:
                    if bullet.strip():
                        md_parts.append(f"- {bullet.strip().lstrip('- ')}")
        
        return "\n".join(md_parts)

    def generate_ksc_response(self, ksc_question: str, user_profile: Dict, experiences: List[Dict[str, Any]], company_intel: Dict[str, str], role_title: str) -> Dict[str, Any]:
        """Generates a KSC response and returns a dictionary with content."""
        relevant_experiences = find_relevant_experiences(ksc_question, experiences)
        experience_text = "\n\n".join([f"Title: {exp['title']}\nSituation: {exp['situation']}\nTask: {exp['task']}\nAction: {exp['action']}\nResult: {exp['result']}" for exp in relevant_experiences])
        
        prompt = f"""
        **Persona:** You are an expert career coach for the Australian Community Services sector. Your tone is professional and authentic, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'N/A')}
        **Task:** Write a compelling KSC response to the question below.
        **KSC Question:** "{ksc_question}"
        **Reasoning Framework:**
        1.  **Deconstruct:** Identify the core competency in the KSC question.
        2.  **Select Evidence:** Choose the strongest parts of the provided STAR stories to prove this competency.
        3.  **Draft:** Structure the response using the STAR method. Weave in the user's personal style and align with the company's values.
        **Candidate's Most Relevant Career Examples:**
        ---
        {experience_text if experience_text else "No specific examples provided."}
        ---
        **Output Format:** Generate clean, professional Markdown. Start directly with the response, do not add extra headings like "KSC Response".
        """
        markdown_content = self._generate_ai_content(prompt)
        return {"html": markdown_content}

    def generate_cover_letter_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]], job_details: Dict[str, Any], company_intel: Dict[str, Any]) -> str:
        """Generates a cover letter as a Markdown string."""
        job_desc_for_search = job_details.get('full_text', '')
        most_relevant_experience = find_relevant_experiences(job_desc_for_search, experiences, top_k=1)
        experience_snippet = ""
        if most_relevant_experience:
            exp = most_relevant_experience[0]
            experience_snippet = f"For instance, in my role as a {exp['title']} at {exp['company']}, I was responsible for {exp['task']}. I successfully {exp['action']}, which directly resulted in {exp['result']}."
        
        prompt = f"""
        **Persona:** You are an expert career advisor writing a cover letter for the Australian Community Services sector. Your tone is professional and warm, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'I am deeply impressed by your commitment to the community.')}
        **Task:** Write a compelling three-paragraph cover letter.
        **Reasoning Framework:**
        1.  **Opening:** State the role you are applying for and express genuine enthusiasm for the company, referencing a specific piece of company intelligence.
        2.  **Body:** Connect your skills directly to the key requirements of the job. Integrate the "Most Relevant Career Example" to provide concrete proof of your abilities and achievements.
        3.  **Closing:** Reiterate your strong interest in the role and the company. Include a clear call to action, stating your eagerness to discuss your application further.
        **Most Relevant Career Example:**
        ---
        {experience_snippet if experience_snippet else "The applicant has extensive experience directly relevant to this role's requirements."}
        ---
        **Output Format:** Generate clean, professional Markdown for a cover letter.
        """
        return self._generate_ai_content(prompt)

    def score_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Scores a resume against a job description using AI."""
        prompt = f"""
        **Persona:** You are an expert ATS (Applicant Tracking System) and a senior recruiter for the Community Services sector.
        **Task:** Analyze the provided resume against the job description. Provide a match score and actionable feedback.
        **Reasoning Framework:**
        1.  **Keyword Analysis:** Extract key skills, qualifications, and duties from the job description.
        2.  **Resume Parsing:** Identify skills, experiences, and achievements in the resume.
        3.  **Alignment Scoring:** Calculate a percentage score based on how well the resume matches the key requirements. Score harshly and realistically. A perfect match is rare.
        4.  **Feedback Generation:** Provide a list of strengths (what matched well) and a list of concrete, actionable suggestions for improvement (e.g., "Add the keyword 'Child Safety Framework' to your skills section," or "Quantify the achievement in your role at Hope Services by mentioning the number of clients served.").
        
        **Job Description:**
        ---
        {job_description}
        ---
        
        **Candidate's Resume:**
        ---
        {resume_text}
        ---
        
        **Output Format:**
        Return a single, valid JSON object only. Do not include any other text.
        {{
          "match_score": <integer_percentage>,
          "strengths": ["...", "..."],
          "suggestions": ["...", "..."]
        }}
        """
        response_text = self._generate_ai_content(prompt)
        try:
            # Clean up potential markdown formatting around the JSON
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Could not parse AI response.", "raw_text": response_text}
