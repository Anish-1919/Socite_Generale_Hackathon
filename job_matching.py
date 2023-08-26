import streamlit as st
import pdfplumber
import docx2txt
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Job Matching",
    page_icon=":student:",  # You can use an emoji or a URL to an image
)


def extract_skills(text):
    # Define a list of common skills (you can expand this list)
    skills_list = ['Proficiency Java', 'content creation', 'Financial leadership', 'User-centered design', 
                   'financial reporting', 'hardware/software assistance', 'cybersecurity', 'issue resolution', 
                   'Network security', 'Medical assessment', 'problem-solving', 'reporting', 'Environmental regulations', 
                   'Accounting principles', 'Process optimization', 'maintenance', 'Medical degree', 'Environmental analysis', 
                   'child health', 'Recruitment', 'employee relations', 'market analysis', 'budgeting', 'mechanical design', 
                   'testing methodologies', 'Occupational health', 'team management', 'Architectural design', 'Logistics', 
                   'Creative direction', 'HR operations', 'Pharmacology', 'Space planning', 'Quality assurance', 
                   'Food product development', 'classroom management', 'patient counseling', 'oral health education', 
                   'leadership', 'patient assessment', 'Marketing campaigns', 'Social media strategy', 'litigation support', 
                   'software architecture', 'recruitment', 'strategic planning', 'User research', 'technical support', 
                   'rehabilitation', 'regulations', 'retirement planning', 'Electrical systems', 'Rehabilitation techniques', 
                   'Legal documentation', 'inspections', 'wireframing', 'user testing', 'Financial modeling', 'Therapy support', 
                   'curriculum development', 'case analysis', 'incident response', 'supply chain', 'CRM', 'user manuals', 
                   'Therapy techniques', 'Java', 'aerodynamics', 'labor support', 'intrusion detection', 'client acquisition', 
                   'Midwifery knowledge', 'Culinary skills', 'Copywriting', 'Network maintenance', 'investment strategies', 
                   'Digital marketing', 'case management', 'lesson planning', 'storytelling', 'data collection', 'client consultation',
                   'requirements gathering', 'typography', 'scheduling', 'Web design tools', 'thermodynamics', 'Equipment maintenance', 
                   'Data pipeline development', 'communication', 'menu planning', 'data preprocessing', 'document review', 
                   'CSS', 'forecasting', 'negotiation', 'scalability', 'Technical documentation', 'Employee relations', 
                   'Sustainability strategies', 'document preparation', 'Epidemiological research', 'sales techniques', 
                   'Cybersecurity', 'repairs', 'medical knowledge', 'research methodologies', 'Technology solutions', 
                   'Product development', 'Legal consultation', 'Project planning', 'Aerospace systems', 'HTML', 'Communication', 
                   'consumer insights', 'design coordination', 'Healthcare management', 'vendor management', 'QA processes', 
                   'patient assistance', 'maternal care', 'contract drafting', 'exercise programming', 'quality control', 'team leadership', 
                   'Python', 'troubleshooting', 'visual communication', 'wiring', 'environmental impact', 'laboratory techniques', 
                   'Civil engineering principles', 'System design', 'AutoCAD', 'patient care', 'Workplace health programs', 
                   'construction support', 'teamwork', 'Technical training', 'risk management', 'Financial oversight', 
                   'System maintenance', 'Culinary expertise', 'Social media marketing', 'data visualization', 'inventory management', 
                   'empathy', 'aesthetics', 'Research methodologies', 'Data analysis', 'UI/UX principles', 'architectural drawings', 
                   'medication dispensing', 'prototyping', 'Chemical analysis', 'responsive design', 'Event coordination', 
                   'Financial analysis', 'diagnostics', 'Personal training', 'research','HR policies', 'design principles', 
                   'project management', 'menu creation', 'Medication dispensing', 'C++', 'circuit analysis', 'machine learning algorithms', 
                   'Technical support', 'Legal research', 'nutrition', 'Legal expertise', 'PCB design', 'Social support', 
                   'Advanced nursing degree', 'brand messaging', 'Adobe Creative Suite', 'JavaScript', 'talent acquisition', 
                   'documentation', 'Network configuration', 'benefits', 'database management', 'Subject expertise', 'health assessments', 
                   'investments', 'publication', 'analytics', 'Aerospace design', 'data analysis', 'Excel', 'testing', 'bug tracking', 
                   'Project management', 'Nursing degree', 'food safety', 'Electrical design', 'sterilization', 'Operations analysis', 
                   'SQL', 'campaign management', 'collaboration', 'process optimization', 'networking', 'infrastructure', 'Sales strategies', 
                   'advisory services', 'social media', 'compliance', 'SEO', 'visual design', 'threat detection', 'counseling', 
                   'stakeholder coordination', 'vendor relations', 'client communication', 'Financial planning', 'Market analysis', 
                   'operations oversight', 'Event planning', 'debugging', 'Architectural software', 'CAD software', 'Content planning', 
                   'software development', 'propulsion systems', 'materials science', 'ergonomic assessment', 'kitchen management',
                    'Workplace health', 'patient support', 'Media relations', 'logistics', 'Dental procedures', 'event planning', 
                    'structural design', 'Troubleshooting', 'prescription processing', 'usability testing', 'employee onboarding',
                    "Java", "Data Analysis", "Machine Learning", "Communication", "Problem Solving", "Teamwork","data analytics"]
    
    
    # Extract skills from the text using regex
    extracted_skills = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
    return extracted_skills

def main():
    st.title("Upload Resume ")
    model = joblib.load('resume_matching_model.pkl')
    vocabulary = joblib.load('tfidf_vocabulary.pkl')
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

    if uploaded_file is not None:
        content = uploaded_file.read()

        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(uploaded_file)

        st.write("Extracted Text:")
        # st.text(text)
        skills_input=''
        skills = extract_skills(text)
        print(skills)
        if skills:
            for skill in skills:
                skills_input+=skill+','
        else:
            st.write("Please upload resume with skills ")
        tfidf_vectorizer = TfidfVectorizer(max_features=5)
        if skills_input:
            skills = [skill.strip() for skill in skills_input.split(',')]
            skills_text = ' '.join(skills)
            skills_tfidf = tfidf_vectorizer.fit_transform([skills_text])

            # Predict job category
            prediction = model.predict(skills_tfidf.toarray())[0]
            st.write("Predicted Job Category:", prediction)

if __name__ == "__main__":
    main()