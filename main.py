from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Lee Kuan Yew[b] (born Harry Lee Kuan Yew; 16 September 1923 – 23 March 2015), also known as LKY, was a Singaporean politician and barrister who served as the first prime minister of Singapore from 1959 to 1990. A member of the People's Action Party (PAP), his authoritarian governance oversaw the transformation of post-independence Singapore into a highly developed country and one of the four Asian Tigers. He is regarded as the founding father of modern Singapore.
    """
    print(information)

    summary_template = """
    Given the following information, provide a concise summary in 2-3 sentences:
    {information}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    # model = GoogleGenerativeAI(model="gemini-3.8-flash", api_key=api_key, temperature=0)
    model = ChatOllama(temperature=0, model ="gpt-oss:latest")
    chain = summary_prompt_template | model
    response = chain.invoke(input={"information": information})
    print("----")
    print("Generated Summary:", response)


if __name__ == "__main__":
    main()