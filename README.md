# 🤖 Resume Agent

Resume Agent is a powerful, AI-driven Streamlit application designed to help you create professional and tailored job application documents, including resumes, cover letters, and Key Selection Criteria (KSC) responses.

## ✨ Features

- **AI-Powered Content Generation:** Leverages Google's Gemini API to craft compelling content for your application documents.
- **Dynamic Document Creation:** Generate tailored documents on-the-fly based on your personal profile, career history, and specific job descriptions.
- **Company Intelligence:** Uses the Perplexity API to gather real-time insights about a company's values, mission, and recent news to further customize your application.
- **Semantic Caching:** To improve speed and reduce costs, the agent uses an intelligent caching layer. Instead of re-running API calls for queries that are slightly different but conceptually the same, it uses vector embeddings to find and return semantically similar cached results.
- **Multiple Document Types:** Supports the creation of:
    - Resumes
    - Cover Letters
    - Key Selection Criteria (KSC) responses
- **Local Data Storage:** Securely stores your user profile and career history in a local SQLite database.
- **Flexible Export Options:** Download your generated documents as either DOCX or PDF files.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **AI Models:** Google Gemini, Perplexity Sonar
- **Vector Embeddings:** `sentence-transformers`
- **Database:** SQLite
- **Document Processing:** `python-docx`, `weasyprint`, `pypdf`

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.8+
- A Git client

### 2. Installation

First, clone the repository to your local machine:

```bash
git clone <your-repository-url>
cd resume-agent
3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

Bash



# Create a virtual environment
python3 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
4. Install Dependencies
Install all the required packages using the requirements.txt file:

Bash



pip install -r requirements.txt
5. Configuration
This application requires API keys for Google Gemini and Perplexity.

Obtain your API keys from:

Google AI Studio for the Gemini key.

Perplexity Labs for the Perplexity key.

When you run the app, navigate to the Settings page in the UI to enter and save your API keys. For deployment, these should be set as secrets.

▶️ Usage
To run the application locally, use the following command from the project's root directory:

Bash



streamlit run main_app.py
Your web browser will open with the application running.

☁️ Deployment
This application is ready to be deployed to Streamlit Community Cloud.

Push your code to a GitHub repository.

On Streamlit Community Cloud, link your GitHub account and select the repository.

Ensure the main file path is set to main_app.py.

In the advanced settings, add your API keys as Secrets:

Ini, TOML


gemini_api_key = "YOUR_GEMINI_API_KEY"
perplexity_api_key = "YOUR_PERPLEXITY_API_KEY"
Click Deploy!
