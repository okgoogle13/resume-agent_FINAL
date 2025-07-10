# pages/1_Manage_Career_History.py
import streamlit as st
import database as db

st.set_page_config(page_title="Manage Career History", layout="wide")
st.title("📝 Manage Career History")
st.write("Add, edit, or delete your career examples here. These examples, including your 'gold standard' resume bullet points, will be used by the AI to tailor your job applications.")

st.header("Add or Edit Experience")
query_params = st.experimental_get_query_params()
edit_id = query_params.get("edit", [None])[0]
initial_data = {}
if edit_id:
    initial_data = db.get_experience_by_id(int(edit_id))
    if not initial_data:
        st.error("Experience not found."); edit_id = None

with st.form(key="experience_form", clear_on_submit=not edit_id):
    col1, col2, col3 = st.columns(3)
    with col1: title = st.text_input("Job Title", value=initial_data.get("title", ""), placeholder="e.g., Community Support Worker")
    with col2: company = st.text_input("Company / Organization", value=initial_data.get("company", ""), placeholder="e.g., Hope Services")
    with col3: dates = st.text_input("Dates of Employment", value=initial_data.get("dates", ""), placeholder="e.g., Jan 2022 - Present")
    st.subheader("STAR Method Example")
    situation = st.text_area("Situation", value=initial_data.get("situation", ""), placeholder="Describe the context or background.")
    task = st.text_area("Task", value=initial_data.get("task", ""), placeholder="What was your specific goal or responsibility?")
    action = st.text_area("Action", value=initial_data.get("action", ""), placeholder="What steps did you take?")
    result = st.text_area("Result", value=initial_data.get("result", ""), placeholder="What was the outcome? Use quantifiable data if possible.")
    skills = st.text_input("Related Skills (comma-separated)", value=initial_data.get("related_skills", ""), placeholder="e.g., crisis-intervention, client-advocacy")
    st.markdown("---")
    st.subheader("Gold Standard Resume Bullet Points")
    st.info("Add your best, pre-written resume bullet points for this experience (one per line). The AI will use these to build your resume.")
    resume_bullets = st.text_area("Resume Bullet Points", value=initial_data.get("resume_bullets", ""), height=150, placeholder="e.g., Achieved X by doing Y, resulting in Z.")
    submit_button = st.form_submit_button(label="Save Experience" if not edit_id else "Update Experience")
    if submit_button:
        if not all([title, company, dates, situation, task, action, result]):
            st.warning("Please fill out all fields.")
        else:
            if edit_id:
                db.update_experience(int(edit_id), title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience updated successfully!"); st.experimental_set_query_params()
            else:
                db.add_experience(title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience added successfully!")
            st.experimental_rerun()

st.header("Your Saved Experiences")
all_experiences = db.get_all_experiences()
if not all_experiences:
    st.info("You haven't added any experiences yet. Use the form above to get started.")
else:
    for exp in all_experiences:
        with st.expander(f"**{exp['title']} at {exp['company']}** (ID: {exp['id']})"):
            st.markdown(f"**Dates:** {exp['dates']}"); st.markdown(f"**Situation:** {exp['situation']}")
            st.markdown(f"**Task:** {exp['task']}"); st.markdown(f"**Action:** {exp['action']}")
            st.markdown(f"**Result:** {exp['result']}"); st.markdown(f"**Skills:** `{exp['related_skills']}`")
            if exp.get('resume_bullets'):
                st.markdown("**Resume Bullets:**"); st.code(exp['resume_bullets'], language='text')
            col1, col2 = st.columns([0.1, 1])
            with col1:
                if st.button("Edit", key=f"edit_{exp['id']}"):
                    st.experimental_set_query_params(edit=exp['id']); st.experimental_rerun()
            with col2:
                if st.button("Delete", key=f"delete_{exp['id']}", type="primary"):
                    db.delete_experience(exp['id']); st.experimental_rerun()
