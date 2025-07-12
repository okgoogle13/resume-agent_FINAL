# pages/0_User_Profile.py
import streamlit as st
import database as db
from config import USER_PROFILE_TITLE, DEFAULT_USER_PROFILE

st.set_page_config(page_title=USER_PROFILE_TITLE, layout="wide")
st.title(f"👤 {USER_PROFILE_TITLE}")
st.write("This information will be used to populate your documents. It is saved locally in your `career_history.db` file.")

profile = db.get_user_profile() or {}

with st.form(key="profile_form"):
    st.header("Contact Information")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", value=profile.get("full_name", DEFAULT_USER_PROFILE["full_name"]))
        email = st.text_input("Email Address", value=profile.get("email", DEFAULT_USER_PROFILE["email"]))
        phone = st.text_input("Phone Number", value=profile.get("phone", DEFAULT_USER_PROFILE["phone"]))
    with col2:
        address = st.text_input("Address", value=profile.get("address", DEFAULT_USER_PROFILE["address"]))
        linkedin_url = st.text_input("LinkedIn Profile URL", value=profile.get("linkedin_url", DEFAULT_USER_PROFILE["linkedin_url"]))
    st.header("Professional Summary")
    professional_summary = st.text_area("Summary / Personal Statement", value=profile.get("professional_summary", ""), height=150, placeholder="Write a brief 2-4 sentence summary of your career, skills, and goals.")
    
    style_profile_text = profile.get("style_profile", "")

    submit_button = st.form_submit_button("Save Profile")
    if submit_button:
        profile_data = {"full_name": full_name, "email": email, "phone": phone, "address": address, "linkedin_url": linkedin_url, "professional_summary": professional_summary, "style_profile": style_profile_text}
        db.save_user_profile(profile_data)
        st.toast("✅ Profile saved successfully!")
