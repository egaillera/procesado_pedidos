import json
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputToolsParser, JsonOutputKeyToolsParser
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

from pydantic import BaseModel, Field
from typing import List,Optional, Literal, Set
import pandas as pd
import math
import os

from dotenv import load_dotenv

from constants import PATH_TO_CATALOG
from extraction_schema import get_examples

def read_products(excel_file):
    df = pd.read_excel(excel_file)
    return ";".join(list(df["DESCRIPCION"]))


def create_extraction_agent():

    load_dotenv()

    products_list = read_products(PATH_TO_CATALOG)

    from extraction_schema import Product, Information

    functions = [Information]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    # TODO: get ready to use it with Mistral. Doesn't work now
    #llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    llm_with_functions = llm.bind_tools(functions)


    system_prompt = "You are an expert extraction algorithm. \
                    Only extract relevant information from the text \
                    If you do not know the value ofan attribute asked to extract, \
                    return null for the attribute's value."
                    # Take into account the following list of products, separated by ';': " + products_list 

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
         MessagesPlaceholder('examples'),
         ("user", "{input}"),
        ])
    
    # TO USE WITH OPENAI
    extraction_chain = prompt | llm_with_functions | JsonOutputKeyToolsParser(key_name="Information")
    
    '''
    extraction_chain = prompt | llm_with_functions.with_structured_output(
        schema = Information,
        method = "function_calling",
        include_raw = False
    ) '''

    # TO USE WITH MISTRAL
    #extraction_chain = prompt | llm_with_functions | PydanticToolsParser(tools=[Information]) 
    
    return extraction_chain

def execute_extractor(request:str):
    load_dotenv()
    agent = create_extraction_agent()
    examples = get_examples()
    result = agent.invoke({"input":request,"examples":examples})

    return result

def main():

    while True:
        purchase = input("Escribe tu pedido: ")
        result = execute_extractor(purchase)
        print(result)


if __name__ == "__main__":
    main()


