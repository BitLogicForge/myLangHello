from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello, myLang!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    template = "What is a good name for a company that makes {product}?"
    prompt = PromptTemplate(input_variables=["product"], template=template)
    product = "colorful socks"
    chain = prompt | llm

    response = chain.invoke({"product": product})

    # Print only the content of the response for human readability
    print("Response from LLM:")
    if hasattr(response, "content"):
        print(response.content)
        print("--- Full Response Object ---")
        # format kwargs to nice format
        print(response.additional_kwargs)
        print("--- Response Metadata ---")
        print(response.response_metadata)

    else:
        print(response)


if __name__ == "__main__":
    main()
