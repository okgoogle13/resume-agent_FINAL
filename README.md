# 🤖 Resume Agent

Resume Agent is a powerful, AI-driven Streamlit application designed to help you create professional and tailored job application documents, including resumes, cover letters, and Key Selection Criteria (KSC) responses.

## ✨ Features

- **AI-Powered Content Generation:** Leverages Google's Gemini API to craft compelling content for your application documents.
- **Dynamic Document Creation:** Generate tailored documents on-the-fly based on your personal profile, career history, and specific job descriptions.
- **Company Intelligence:** Uses the Perplexity API to gather real-time insights about a company's values, mission, and recent news to further customize your application.
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
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Configuration

This application requires API keys for Google Gemini and Perplexity.

1.  Obtain your API keys from:
    - [Google AI Studio](https://aistudio.google.com/app/apikey) for the Gemini key.
    - [Perplexity Labs](https://www.perplexity.ai/settings/api) for the Perplexity key.
2.  When you run the app, navigate to the **Settings** page in the UI to enter and save your API keys. For deployment, these should be set as secrets.

## ▶️ Usage

To run the application locally, use the following command from the project's root directory:

```bash
streamlit run main_app.py
```

Your web browser will open with the application running.

## ☁️ Deployment

This application is ready to be deployed to [Streamlit Community Cloud](https://share.streamlit.io).

1.  Push your code to a GitHub repository.
2.  On Streamlit Community Cloud, link your GitHub account and select the repository.
3.  Ensure the main file path is set to `main_app.py`.
4.  In the advanced settings, add your API keys as **Secrets**:
    ```toml
    gemini_api_key = "YOUR_GEMINI_API_KEY"
    perplexity_api_key = "YOUR_PERPLEXITY_API_KEY"
    ```
5.  Click **Deploy!**
