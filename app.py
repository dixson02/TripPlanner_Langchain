import streamlit as st
from typing import Optional, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models import LLM
import google.generativeai as genai

# --- Configure Gemini ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def generate_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    story = []

    for line in text.split('\n'):
        if line.strip():  # skip empty lines
            para = Paragraph(line.strip(), styles["Normal"])
            story.append(para)
            story.append(Spacer(1, 10))  # space between lines

    doc.build(story)
    buffer.seek(0)
    return buffer



class GeminiLLM(LLM):
    model: any = genai.GenerativeModel()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
    @property
    def _llm_type(self) -> str:
        return "gemini"

# --- Prompt Template ---
trip_prompt = PromptTemplate(
    input_variables=[
        "destination", 
        "travel_dates", 
        "trip_duration", 
        "budget", 
        "travel_style", 
        "interests", 
        "travelers", 
        "accommodation_type"
    ],
    template="""
You are a professional travel planner.

Create a personalized {trip_duration}-day itinerary for a trip to {destination} for {travelers} during {travel_dates}.
The budget is {budget}, and the traveler(s) prefer {travel_style} travel with interests including: {interests}.
They prefer to stay in {accommodation_type}.

Please include the following in the trip plan:
- A daily itinerary with morning, afternoon, and evening activities
- Suggested places to eat and visit each day
- Transportation tips
- Local experiences or hidden gems
- Any travel advisories or tips

Format the response in a clean, organized itinerary. Be specific and practical.
"""
)

# --- LangChain Setup ---
llm = GeminiLLM()
llm_chain = LLMChain(prompt=trip_prompt, llm=llm)

# --- Streamlit UI Config ---
st.set_page_config(page_title="🌍 AI Trip Planner", page_icon="🧳", layout="centered")

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🌍 Plan Your Dream Trip with AI</h1>", unsafe_allow_html=True)
st.markdown("Fill in your preferences and let AI create a tailored travel itinerary just for you! ✈️")

# --- Optional Banner Image ---
st.image("images.png", use_container_width=True)

# --- Input Fields Layout ---
st.markdown("## ✏️ Trip Details")

col1, col2 = st.columns(2)

with col1:
    destination = st.text_input("📍 Destination", "India")
    travel_dates = st.text_input("📅 Travel Dates", "September 10–20, 2025")
    trip_duration = st.number_input("📆 Trip Duration (days)", min_value=1, max_value=30, value=10)
    travelers = st.text_input("🧑‍🤝‍🧑 Who is traveling?", "a couple")

with col2:
    budget = st.text_input("💰 Budget in INR",1000)
    travel_style = st.selectbox("🧳 Travel Style", ["Relaxed", "Adventure", "Cultural", "Luxury", "Backpacking", "Scenic"])
    interests = st.text_area("🎯 Interests", "temples, nature, food, anime, local markets")
    accommodation_type = st.selectbox("🏨 Accommodation Type", [
        "Budget hostel", "Mid-range hotel", "Luxury resort", "Traditional stays (e.g., ryokan)", "Airbnb"
    ])

# --- Generate Button ---
st.markdown("### 🛫 Ready to Plan?")
if st.button('✨ Create My Trip Plan'):
    if all([destination, travel_dates, trip_duration, budget, travel_style, interests, travelers, accommodation_type]):
        input_dict = {
            "destination": destination,
            "travel_dates": travel_dates,
            "trip_duration": str(trip_duration),
            "budget": budget,
            "travel_style": travel_style,
            "interests": interests,
            "travelers": travelers,
            "accommodation_type": accommodation_type
        }
        with st.spinner("Generating your travel itinerary..."):
            res = llm_chain.run(input_dict)
        st.success("✅ Trip Plan Ready!")
        st.markdown("### 📋 Your Personalized Itinerary")
        st.markdown(res)
                # PDF Download
        pdf_buffer = generate_pdf(res)
        st.download_button(
            label="📄 Download Trip Plan as PDF",
            data=pdf_buffer,
            file_name="trip_plan.pdf",
            mime="application/pdf"
        )

    else:
        st.warning("🚨 Please fill in all fields before generating your trip plan.")
