from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional,List
from langchain_core.language_models import LLM
# from api_keys import GEMINI_API_KEY
import google.generativeai as genai
import streamlit as st
GEMINI_API_KEY=st.st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
class GeminiLLM(LLM):
    model: any = genai.GenerativeModel()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
    @property
    def _llm_type(self) -> str:
        return "gemini"



from langchain.prompts import PromptTemplate

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

# from model_and_prompt import trip_prompt,GeminiLLM
# import streamlit as st

# Now you can use it in LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
llm = GeminiLLM()
llm_chain = LLMChain(prompt=trip_prompt, llm=llm)
st.title('Trip Planner In Just 2 Minutes')
if 'history' not in st.session_state:
    st.session_state['history']=[]
st.image('images.png')

destination = st.text_input("Destination", "India")
travel_dates = st.text_input("Travel Dates", "September 10–20, 2025")
trip_duration = st.number_input("Trip Duration (in days)", min_value=1, max_value=30, value=10)
budget = st.selectbox("Budget", ["Low (<$1000)", "Medium ($1000–$3000)", "High (>$3000)"])
travel_style = st.selectbox("Travel Style", ["Relaxed", "Adventure", "Cultural", "Luxury", "Backpacking", "Scenic"])
interests = st.text_area("Interests (comma separated)", "temples, nature, food, anime, local markets")
travelers = st.text_input("Who is traveling?", "a couple")
accommodation_type = st.selectbox("Accommodation Type", ["Budget hostel", "Mid-range hotel", "Luxury resort", "Traditional stays (e.g., ryokan)", "Airbnb"])
# button=st.button('Create Plan')
if st.button('Create Plan'):
    if all([destination, travel_dates, trip_duration, budget, travel_style, interests, travelers, accommodation_type]):
        input_dict = {
            "destination": destination,
            "travel_dates": travel_dates,
            "trip_duration": str(trip_duration),  # convert to string if template expects string
            "budget": budget,
            "travel_style": travel_style,
            "interests": interests,
            "travelers": travelers,
            "accommodation_type": accommodation_type
        }
        res = llm_chain.run(input_dict)
        st.write(res)
    else:
        st.warning("Please fill in all fields.")