import json
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import PydanticToolsParser
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

from pydantic import BaseModel, Field
from typing import List,Optional, Literal, Set
import pandas as pd
import math

import os

from dotenv import load_dotenv

# Features of products
categories = []
qualities = []
weights = ["100 gramos","250 gramos","400 gramos","500 gramos","750 gramos","1 kilogramo", "2 kilogramos"]
formats = []
tastes = []

path_to_catalog = "./CatalogoEstructurado.xlsx"


def read_products(excel_file):
    global categories
    global qualities
    global formats
    global tastes

    df = pd.read_excel(excel_file)

    # Convert to lowercase and avoid NaNs
    categories = [c.lower() for c in list(set(df["TIPO"])) 
                  if not (isinstance(c,float) and math.isnan(c))]
    qualities = [q.lower() for q in list(set(df["CALIDAD"])) 
                 if not (isinstance(q,float) and math.isnan(q))]
    formats = [f.lower() for f in list(set(df["FORMATO"]))  
               if not (isinstance(f,float) and math.isnan(f)) ]
    tastes = [t.lower() for t in list(set(df["GUSTO"]))    
              if not (isinstance(t,float) and math.isnan(t))]

    return ";".join(list(df["DESCRIPCION"]))


def create_extraction_agent():

    load_dotenv()

    products_list = read_products(path_to_catalog)

    class Product(BaseModel):
        """Information about a product"""
        name_by_client: str = Field(description="name used by the customer to refer this product")
        amount: str = Field(description="number of units of this product in the purchase order. Could be 1/2 as well",examples=["1","1/2","2","3"])
        type: Literal[*categories] = Field(description="type of product", examples=categories) # type: ignore
        quality: Optional[str] = Field(description="adjective that indicates quality of the product",
                                       examples=qualities)
        weight: Optional[str] = Field(description="Weight of the product", examples=weights)
        format: Optional[str] = Field(description="Format, which can describe both the type of cut but also the packaging",
                                      examples=formats) 
        taste: Optional[str] = Field(description="What is the taste of the product",examples=tastes)


    class Information(BaseModel):
        """Information to extract from the purchase order"""
        products: List[Product] = Field(description="list of products of the purchase order")

    functions = [Information]

    # TO USE WITH OPENAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    llm_with_functions = llm.bind_functions(functions,function_call={"name":"Information"})

    # TO USE WITH MISTRAL
    #llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    #llm_with_functions = llm.bind_tools(functions)

    system_prompt = "Think carefully and then extract the list of products of the purchase order, \
       taking into account this JSON catalogue: " + products_list

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
        ])
    
    # TO USE WITH OPENAI
    extraction_chain = prompt | llm_with_functions | JsonOutputFunctionsParser()

    # TO USE WITH MISTRAL
    #extraction_chain = prompt | llm_with_functions | PydanticToolsParser(tools=[Information]) 
    
    return extraction_chain

def main():
    load_dotenv()

    agent = create_extraction_agent()

    while True:
        purchase = input("Escribe tu pedido: ")
        result = agent.invoke({"input":purchase})
        print(result)


if __name__ == "__main__":
    main()


