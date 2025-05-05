import google.generativeai as gemini
import json
from datasets import load_dataset

from dense import dense_search
from hybrid import hybrid_search

from athina_tools import upload_dataset

dataset = load_dataset("PatronusAI/financebench", split="train")

# Configure API key
gemini.configure(api_key="GEMINI_API_KEY")  # Replace with your actual API key

# Initialize the Gemini 1.5 Flash model
model = gemini.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(system_prompt, user_prompt):
    """
    Generates a response from the Gemini model, using a system prompt and a user prompt.

    Args:
        system_prompt: The system prompt (string).
        user_prompt: The user's prompt (string).

    Returns:
        The generated text response (string) or None if an error occurs.
    """
    try:
        contents = [
            {
                "role": "user",
                "parts": [system_prompt]
            },
            {
                "role": "user",
                "parts": [user_prompt]
            }
        ]

        response = model.generate_content(contents)
        # print("response text: ", response.text)
        return response.text  # Extract text from the response

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return None

def format_response(text):
    d = json.loads(text)
    return d

docs = ['AMCOR_2022_8K_dated-2022-07-01', 'ADOBE_2017_10K', 'ACTIVISIONBLIZZARD_2019_10K', 'AES_2022_10K', 'AMAZON_2019_10K', '3M_2018_10K', '3M_2023Q2_10Q', 'AMCOR_2020_10K', '3M_2022_10K', 'AMAZON_2017_10K', 'ADOBE_2015_10K', 'ADOBE_2016_10K', "ADOBE_2022_10K"]
#docs = ["ADOBE_2017_10K"]

def get_answer(question, results):
    system_prompt_answer = """
    You are an expert question-answering system. Your task is to provide accurate and concise answers to user questions based solely on the information provided in the retrieved documents.
    You must adhere to the following guidelines:

    1.  **Strictly Answer from Context:** Only use information present in the retrieved documents to formulate your response. Do not invent or infer information beyond what is explicitly stated in the provided context.
    2.  **Concise and Direct Answers:** Provide direct and to-the-point answers. Avoid unnecessary elaboration or extraneous details.
    3.  **Acknowledge Limitations:** If the retrieved documents do not contain the answer to the user's question, state clearly that the answer cannot be found in the provided context. Do not attempt to guess or provide speculative answers.
    4.  **Maintain Factual Accuracy:** Ensure that your responses are factually accurate and consistent with the information presented in the retrieved documents.
    5.  **Use the Language of the Question:** Respond in the same language as the user's question.
    6.  **Avoid General Knowledge:** Do not use any general knowledge that is not present in the provided documents.
    7.  **Answer as if you are reading the provided context for the first time.**
    8.  **When given numerical information, always include the units.**
    9.  **Answer only what is asked do not give any extra information.**

    Your goal is to be a reliable and precise information retriever, focusing on the information given to you.
    """

    user_prompt = f"TOP RAG RESULTS:\n\n{results}\n\n\nUser Question: {question}"

    answer = get_gemini_response(system_prompt_answer, user_prompt)

    return answer

final_data = []

i = 1
cnt = 1
for data in dataset:
    if data["doc_name"] in docs:
        if i==7:
            upload_dataset(f"rag_test_{cnt}", final_data)
            print(f"Uploaded rag_test_{cnt}")

            final_data = []
            cnt+=1
            i = 1

        question = data["question"]
        print(question)
        system_prompt_question = "You are an expert in extracting structured information from text. " \
        "Your task is to identify and extract company name and year, from user-provided questions. " \
        "You must adhere strictly to the requested output format: a plain and normal text dictionary with two keys, company and year." \
        "The dictionary keys and company name should be inside double quotes and year should be an integer." \
        "If information is missing, return an empty string for the corresponding key. " \
        "Do not include any additional text or formatting like ``` or json, etc. Just give a dictionary starting with { and ending with } in plain and normal text." 
        
        prompt = f"""Extract 'company name' and 'year' of the question given below:
        Question: {question}
        Strictly return you answer as a dictionary IN PLAIN TEXT containing two keys: company and year enclosed in double quotes.
        If you cannot find anyone of them, simply return empty string for that key's value.
        DO NOT RETURN ANY OTHER TEXT LIKE ``` or json, JUST A DICTIONARY STARTING WITH {{ and ending with }} IN PLAIN TEXT.
        """

        refiners = format_response(get_gemini_response(system_prompt_question, question))
        print(type(refiners))
        print(refiners["company"], type(refiners["company"]))
        print(refiners["year"], type(refiners["year"]))

        company = refiners["company"]
        year = refiners["year"]
        if not refiners["company"]:
            company = data["company"]
        if not refiners["year"]:
            year = data["doc_period"]
        
        # company = data["company"]
        # year = data["doc_period"]

        dense_results = dense_search(question, company, year)
        dense_contents = [content for content,score in dense_results]

        final_data.append({
            "query":question,
            "context":dense_contents,
            "type":"dense"
        })

        # answer = get_answer(question, "".join(dense_contents))
        # print(answer)

        hybrid_contents = hybrid_search(question, company, year)

        final_data.append({
            "query":question,
            "context":hybrid_contents,
            "type":"hybrid"
        })

        # answer = get_answer(question, "".join(hybrid_contents))
        # print(answer)

        i+=1
        print("\n#####################################\n")
    else:
        upload_dataset(f"rag_test_{cnt}", final_data)
        print(f"Uploaded rag_test_{cnt}")
        break