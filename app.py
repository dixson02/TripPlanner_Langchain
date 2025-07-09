from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from model_and_prompt import trip_prompt,GeminiLLM
import streamlit as st

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