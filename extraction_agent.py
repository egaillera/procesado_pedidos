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
from typing import List,Optional, Literal
import pandas as pd

import os

from dotenv import load_dotenv

categories = ["jamon","paleta","presa","chorizo", "morcilla","salchichon","avio","panceta",
             "caldo","crema","pate","queso cabra","queso oveja","queso mezcla oveja y cabra"
            "miel","aceitunas"]

qualities = ["bellota iberico","cebo iberido","cebo campo iberico","bellota iberica 100%",
            "bellota iberica seleccion","100% iberico denominacion de origen","iberico",
            "leche cruda","leche cruda gran reserva","leche cruda ahumado","curado de extremadura serie plata",
            "curado de extremadura serie oro","york"]

weights = ["100 gramos","250 gramos","400 gramos","500 gramos","750 gramos","1 kilogramo", "2 kilogramos"]

formats = ["loncheado","virutas","tarro","embuchado","doblado","cular","herradura"]

tastes = ["atun","finas hierbas","iberico","pimenton","corsevilla","sabor tradicional","picante","blanco","rojo"]




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

    return ";".join(list(df["Descripcion"]))

    '''
    # Initialize the dictionary: one array of products per family
    for family in df["Familia"].unique():
        catalogue[family] = []


    for index, row in df.iterrows():
        catalogue[row['Familia']].append({"Articulo":row['Articulo'],
                                          "Descripcion":row["Descripcion"]})

    json_string = json.dumps(catalogue,indent=4).replace("{", "{{").replace("}", "}}")
    return json_string,vectorstore
    '''



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
        format: Optional[str] = Field(description="Format, which can describe both the type of product cut and the packaging",
                                      examples=formats)
        taste: Optional[str] = Field(description="What is the taste of the product",examples=tastes)


    class Information(BaseModel):
        """Information to extract from the purchase order"""
        products: List[Product] = Field(description="list of products of the purchase order")

    functions = [Information]

    llm = ChatOpenAI(model="gpt-4o", temperature = 0)
    llm_with_functions = llm.bind_functions(functions,function_call={"name":"Information"})

    # TO USE WITH MISTRAL
    #llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    #llm_with_functions = llm.bind_tools(functions)

    '''
    system_prompt = "Think carefully and then extract the products of the purchase order\
                     taking into account this JSON catalogue: " + read_products("./data/Tarifas_por_familia_JyS.xls") \
                     + "\n\n If there is ambiguity or you are not sure about the product or the amount of products, return nothing.\
                        Examples of ambiguity: \
                            quiero un tarro de pate (there are two kinds: PATE DE ATÚN 250 GRS and PATE IBERÍCO 250 GRS)\
                            quiero un loncheado de jamon (there are three: LONCHEADO 100 GR JAMON CURADO C, \
                                                          LONCHEADO 100 GR JAMON DE CEBO IBERICO and LONCHEADO 100 GR JAMON RESERVA)" '''
    '''
    system_prompt = "Think carefully and then extract the number of products of the purchase order. \
                    Also for each product, you must extract: \
                    - the amount of units  (mandatory)\
                    - the type of product, that could be jamon, paleta, lomo, presa, chorizo, morcilla, salchichon, avio, \
                       panceta, caldo, crema, pate, queso cabra, queso oveja, miel, aceitunas, loncheado lomo, loncheado paleta \
                       loncheado jamon, loncheado salchichon (mandatory) \
                    - the quality of the product, that could be iberico, bellota iberico, cebo iberico, cebo campo iberico, \
                        bellota iberica 100%, bellota iberica seleccion, 100% iberico denominacion de origen (optional)"
    '''
    system_prompt = "Think carefully and then extract the list of products of the purchase order, \
        taking into account this JSON catalogue: " + read_products("./data/Tarifas_por_familia_JyS.xls")

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


        '''
        for item in result['products']:
            #print(item["name"])
            result_string += item["name"] + "\n"
            similars = vectorstore.similarity_search_with_relevance_scores(item["name"],k=10)
            for res,score in similars:
                #print(f"* [SIM={score:3f}] {res.page_content}")
                result_string += f"* [SIM={score:3f}] {res.page_content}" + "\n"

        print(result_string)
        '''

if __name__ == "__main__":
    main()


