import google.generativeai as genai
from langchain import PromptTemplate
from config import GEMINI_API_KEY  # Import the API key from config.py

# Configure Google Gemini using the key from config.py
def init_google_gemini():
    genai.configure(api_key=GEMINI_API_KEY)

# Define a function to query Google Gemini
def query_gemini(prompt, model_name="gemini-1.5-flash"):
    # Initialize the Generative Model
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt)
    return response.text.strip()

# Define a prompt template for answering questions
cs_dsa_coding_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI assistant that only answers queries related to Computer Science, "
        "Data Structures and Algorithms (DSA), and coding. "
        "If a question is outside these domains, reply with: "
        "'I can only answer questions related to Computer Science, DSA, and coding.'\n\n"
        "Question: {question}\nAnswer:"
    ),
)

# Define a prompt template for validating topics
validation_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Check if the following text is related to any of the following topics: 'dsa, coding, computer science fundamentals'. "
        "Reply with 'Yes' if it is related to any of these topics, otherwise reply with 'No'.\n\n"
        "Text: {text}\nIs it related to these topics?"
    ),
)

# Custom LangChain-compatible wrapper for Google Gemini
class GoogleGeminiLLM:
    def __init__(self):
        init_google_gemini()  # Initialize the API

    def predict(self, question):
        # Validate the query first
        if not self.is_valid_topic(question):
            return "I can only answer questions related to Computer Science, DSA, and coding."
        
        # Format the initial query prompt
        prompt = cs_dsa_coding_prompt.format(question=question)
        response = query_gemini(prompt)
        
        # Validate the response
        if not self.is_valid_topic(response):
            return "I can only answer questions related to Computer Science, DSA, and coding."

        return response

    def is_valid_topic(self, text):
        # Format the validation prompt
        validation_prompt = validation_prompt_template.format(text=text)
        validation_result = query_gemini(validation_prompt, model_name="gemini-1.5-flash")

        # Check if the result is "Yes"
        return validation_result.lower() == "yes"


# Usage
if __name__ == "__main__":
    gemini_llm = GoogleGeminiLLM()

    while True:
        user_query = input("Ask your question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        try:
            print("AI Response:", gemini_llm.predict(user_query))
        except Exception as e:
            print(f"An error occurred: {e}")
