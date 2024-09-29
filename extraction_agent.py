from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from pydantic import BaseModel, Field
import pandas as pd

def read_products(excel_file):

    catalogue = ""

    df = pd.read_excel(excel_file)
    for index, row in df.iterrows():
        print(f"Codigo: {format(row['Codigo'],'.0f')}, Descripion: {row['Descripcion']}, Precio: {format(row['Precio'],'.2f')}")
        catalogue = catalogue + f"Codigo: {format(row['Codigo'],'.0f')}, Descripion: {row['Descripcion']}, Precio: {format(row['Precio'],'.2f')}" + "\n"

    return catalogue




def main():

    print(read_products("./data/Tarifas.xls"))

if __name__ == "__main__":
    main()


