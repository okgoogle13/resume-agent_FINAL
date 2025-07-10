# pages/0_User_Profile.py
import streamlit as st
import database as db

st.set_page_config(page_title="User Profile", layout="wide")
st.title("👤 User Profile")
st.write("This information will be used to populate your documents. It is saved locally in your `career_history.db` file.")

profile = db.get_user_profile() or {}

with st.form(key="profile_form"):
    st.header("Contact Information")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", value=profile.get("full_name", "Nishant Jonas Dougall"))
        email = st.text_input("Email Address", value=profile.get("email", ""))
        phone = st.text_input("Phone Number", value=profile.get("phone", "+61412202666"))
    with col2:
        address = st.text_input("Address", value=profile.get("address", "Unit 2 418 high street, Northcote VICTORIA 3070, Australia"))
        linkedin_url = st.text_input("LinkedIn Profile URL", value=profile.get("linkedin_url", ""))
    st.header("Professional Summary")
    professional_summary = st.text_area("Summary / Personal Statement", value=profile.get("professional_summary", ""), height=150, placeholder="Write a brief 2-4 sentence summary of your career, skills, and goals.")
    
    style_profile_text = profile.get("style_profile", "")

    submit_button = st.form_submit_button("Save Profile")
    if submit_button:
        profile_data = {"full_name": full_name, "email": email, "phone": phone, "address": address, "linkedin_url": linkedin_url, "professional_summary": professional_summary, "style_profile": style_profile_text}
        db.save_user_profile(profile_data)
        st.toast("✅ Profile saved successfully!")
