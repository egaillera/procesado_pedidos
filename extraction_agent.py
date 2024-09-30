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
from typing import List
import pandas as pd

import os

from dotenv import load_dotenv



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

    vectorstore = create_vectorstore(df["Descripcion"])

    # Initialize the dictionary: one array of products per family
    for family in df["Familia"].unique():
        catalogue[family] = []


    for index, row in df.iterrows():
        catalogue[row['Familia']].append({"Articulo":row['Articulo'],
                                          "Descripcion":row["Descripcion"], 
                                          "Precio":format(row['Precio'],'.2f')})

    json_string = json.dumps(catalogue,indent=4).replace("{", "{{").replace("}", "}}")
    return json_string,vectorstore


def create_extraction_agent():

    load_dotenv()

    class Product(BaseModel):
        """Information about a product"""
        name: str = Field(description="name used by the customer to refer this product")
        #description: str = Field(description="product description accoding to catalogue")
        #code: int = Field(description="code that identifies the producto")
        amount: int = Field(description="number of units of this product in the purchase order")

    class Information(BaseModel):
        """Information to extract from the purchase order"""
        products: List[Product] = Field(description="list of product of the purchase order")

    functions = [Information]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    llm_with_functions = llm.bind_functions(functions,function_call={"name":"Information"})

    # TO USE WITH MISTRAL
    #llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    #llm_with_functions = llm.bind_tools(functions)

    '''
    system_prompt = "Think carefully and then extract the products of the purchase order\
                     taking into account this JSON catalogue: " + read_products("./data/Tarifas_por_familia_JyS.xls") \
                     + "\n\n If there is ambiguity or you are not sure about the product or the amount of products, return nothing.\
                        Examples of ambiguity: \
                            quiero un tarro de pate (there are two kinds: PATE DE ATÚN 250 GRS and PATE IBERÍCO 250 GRS )\
                            quiero un loncheado de jamon (there are three: LONCHEADO 100 GR JAMON CURADO C, \
                                                          LONCHEADO 100 GR JAMON DE CEBO IBERICO and LONCHEADO 100 GR JAMON RESERVA)" '''
    
    system_prompt = "Think carefully and then extract the products of the purchase order, and the amount of each product"

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

    catalogue, vectorstore = read_products("./data/Tarifas_por_familia_JyS.xls")

    agent = create_extraction_agent()

    while True:
        result_string = ""
        purchase = input("Escribe tu pedido: ")
        result = agent.invoke({"input":purchase})
        #print(result)
        for item in result['products']:
            #print(item["name"])
            result_string += item["name"] + "\n"
            similars = vectorstore.similarity_search_with_relevance_scores(item["name"],k=10)
            for res,score in similars:
                #print(f"* [SIM={score:3f}] {res.page_content}")
                result_string += f"* [SIM={score:3f}] {res.page_content}" + "\n"

        print(result_string)

if __name__ == "__main__":
    main()


