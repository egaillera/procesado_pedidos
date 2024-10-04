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

import os

from dotenv import load_dotenv

categories = ["jamon","paleta","presa","chorizo", "morcilla","salchichon","avio","panceta",
             "caldo","crema","pate","queso cabra","queso oveja","queso mezcla oveja y cabra",
            "miel","aceitunas","other"]

qualities = ["bellota iberico","cebo iberido","cebo campo iberico","bellota iberica 100%",
            "bellota iberica seleccion","100% iberico denominacion de origen","iberico",
            "leche cruda","leche cruda gran reserva","leche cruda ahumado","curado de extremadura serie plata",
            "curado de extremadura serie oro","york","iberico curado"]

weights = ["100 gramos","250 gramos","400 gramos","500 gramos","750 gramos","1 kilogramo", "2 kilogramos"]

formats = ["loncheado","virutas","tarro","embuchado","doblado","cular","herradura","cortado","vela"]

tastes = ["atun","finas hierbas","iberico","pimenton","corsevilla","sabor tradicional","picante","blanco","rojo"]

path_to_catalog = "./data/CatalogoEstructurado.xlsx"




def create_vectorstore(texts):

    embeddings = OpenAIEmbeddings()

    if os.path.exists("./data/vectorstore.index"):
        print("Loading vectorstore ... ")
        vectorstore = FAISS.load_local("./data/vectorstore.index", embeddings,
                                       allow_dangerous_deserialization=True)

    else:
        print("Creating vectorstore")
        vectorstore = FAISS.from_texts(texts = texts, embedding=embeddings)
        vectorstore.save_local("./data/vectorstore.index")

    return vectorstore


def read_products(excel_file):

    catalogue = {}

    df = pd.read_excel(excel_file)

    return ";".join(list(df["DESCRIPCION"]))



def create_extraction_agent():

    load_dotenv()

    class Product(BaseModel):
        """Information about a product"""
        name_by_client: str = Field(description="name used by the customer to refer this product")
        amount: str = Field(description="number of units of this product in the purchase order. Could be 1/2 as well",examples=["1","1/2","2","3"])
        type: Literal[*categories] = Field(description="type of product", examples=categories) # type: ignore
        quality: Optional[str] = Field(description="adjective that indicates quality of the product",
                                       examples=qualities)
        weight: Optional[str] = Field(description="Weight of the product", examples=weights)
        format: Optional[Set[Literal[*formats]]] = Field(description="Format, which can describe both the type of product cut and the packaging", #type: ignore
                                      examples=formats) 
        taste: Optional[str] = Field(description="What is the taste of the product",examples=tastes)


    class Information(BaseModel):
        """Information to extract from the purchase order"""
        products: List[Product] = Field(description="list of products of the purchase order")

    functions = [Information]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    llm_with_functions = llm.bind_functions(functions,function_call={"name":"Information"})

    # TO USE WITH MISTRAL
    #llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    #llm_with_functions = llm.bind_tools(functions)

    system_prompt = "Think carefully and then extract the list of products of the purchase order, \
        taking into account this JSON catalogue: " + read_products(path_to_catalog)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
        ])
    
    extraction_chain = prompt | llm_with_functions | JsonOutputFunctionsParser()

    # TO USE WITH MISTRAL
    #extraction_chain = prompt | llm_with_functions | PydanticToolsParser(tools=[Information])
    
    return extraction_chain

def main():
    load_dotenv()

    #catalogue, vectorstore = read_products("./data/Tarifas_por_familia_JyS.xls")

    agent = create_extraction_agent()

    while True:
        result_string = ""
        purchase = input("Escribe tu pedido: ")
        result = agent.invoke({"input":purchase})
        print(result)


if __name__ == "__main__":
    main()


