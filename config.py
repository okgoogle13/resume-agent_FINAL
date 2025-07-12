# config.py
"""
Central configuration file for the AI Resume Builder application.
This file stores constants, settings, and configurations that are used across the application.
"""

# --- User Information ---
# Default user details that are pre-filled in the User Profile form.
# These can be changed by the user in the UI.
DEFAULT_USER_PROFILE = {
    "full_name": "Nishant Jonas Dougall",
    "email": "nishant.dougall@gmail.com",
    "phone": "+61412202666",
    "address": "Unit 2 418 high street, Northcote VICTORIA 3070, Australia",
    "linkedin_url": "https://www.linkedin.com/in/nishant-dougall-55969322/",
    "professional_summary": ""
}

# --- Document Generation ---
# Placeholder for user name in generated documents. This will be replaced with the user's actual name.
USER_NAME_PLACEHOLDER = "Nishant Jonas Dougall"

# --- Database ---
DB_FILE = "career_history.db"

# --- API Settings ---
# Timeouts and other settings for external API calls.
API_TIMEOUT = 30.0 # seconds

# --- UI Settings ---
# Titles and labels for the Streamlit application.
APP_TITLE = "AI Resume Builder"
USER_PROFILE_TITLE = "User Profile"
CAREER_HISTORY_TITLE = "Manage Career History"
SETTINGS_TITLE = "Settings"
JOB_VAULT_TITLE = "Job Vault"
STYLE_ANALYZER_TITLE = "Style Analyzer"
RESUME_SCORER_TITLE = "Resume Scorer"
