from typing import Optional,List
from langchain_core.language_models import LLM
from api_keys import GEMINI_API_KEY
import google.generativeai as genai
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
