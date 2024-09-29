from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import PydanticToolsParser

from pydantic import BaseModel, Field
from typing import List
import pandas as pd

from dotenv import load_dotenv

def read_products(excel_file):

    catalogue = ""

    df = pd.read_excel(excel_file)
    for index, row in df.iterrows():
        catalogue = catalogue + f"Codigo: {format(row['Codigo'],'.0f')}, \
                                Descripicion: {row['Descripcion']}, \
                                Precio: {format(row['Precio'],'.2f')}" + "\n"

    return catalogue


def create_extraction_agent():

    load_dotenv()

    class Product(BaseModel):
        """Information about a product"""
        description: str = Field(description="descripcion del producto")
        code: int = Field(description="codigo que identifica el producto")
        amount: int = Field(description="number of units of this product in the purchase order")

    class Information(BaseModel):
        """Informatin to extract from the purchase order"""
        products: List[Product] = Field(description="list of product of the purchase order")

    functions = [Information]

    #llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    #llm_with_functions = llm.bind_functions(functions,function_call={"name":"Information"})

    # TO USE WITH MISTRAL
    llm = ChatMistralAI(model="mistral-large-latest",temperature=0)
    llm_with_functions = llm.bind_tools(functions)

    system_prompt = "Think carefully, detect ambiguity and then extract the products of the purchase order\
                     taking into account this catalogue: " + read_products("./data/Tarifas.xls") \
                     + "\n\n If there is ambiguity or you are not sure about the product or the amount of products, return nothing.\
                        Examples of ambiguity: \
                            quiero un tarro de pate (there are two kinds: PATE DE ATÚN 250 GRS and PATE IBERÍCO 250 GRS )\
                            quiero un loncheado de jamon (there are three: LONCHEADO 100 GR JAMON CURADO C, \
                                                          LONCHEADO 100 GR JAMON DE CEBO IBERICO and LONCHEADO 100 GR JAMON RESERVA)"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
        ])
    
    #extraction_chain = prompt | llm_with_functions | JsonOutputFunctionsParser()

    # TO USE WITH MISTRAL
    extraction_chain = prompt | llm_with_functions | PydanticToolsParser(tools=[Information])
    
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


