import uuid
from numpy import Inf
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Set, Dict, TypedDict
import pandas as pd
import math
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


from constants import PATH_TO_CATALOG

df = pd.read_excel(PATH_TO_CATALOG)

# Convert to lowercase and avoid NaNs
categories = [c.lower() for c in list(set(df["TIPO"])) 
                if not (isinstance(c,float) and math.isnan(c))]
categories.append("other") # To avoid the "pizza" effect
qualities = [q.lower() for q in list(set(df["CALIDAD"])) 
                if not (isinstance(q,float) and math.isnan(q))]
weights = ["100 gramos","250 gramos","400 gramos","500 gramos",
           "750 gramos","1 kilogramo", "2 kilogramos"]
formats = [f.lower() for f in list(set(df["FORMATO"]))  
            if not (isinstance(f,float) and math.isnan(f)) ]
tastes = [t.lower() for t in list(set(df["GUSTO"]))    
            if not (isinstance(t,float) and math.isnan(t))]

class Product(BaseModel):
    """Information about a product, including its main features"""
    name_by_client: str = Field(description="name used by the customer to refer this product")
    amount: str = Field(description="number of units of this product in the purchase order. Could be 1/2 as well",examples=["1","1/2","2","3"])
    type: Literal[*categories] = Field(description="type of product", examples=categories) # type: ignore
    quality: Optional[Literal[*qualities]] = Field(description="Adjective that indicates quality of the product") # type: ignore
    weight: Optional[str] = Field(description="Weight of the product", examples=weights)
    format: Optional[Literal[*formats]] = Field(description="Format, which can describe both the type of cut but also the packaging", examples=formats) # type: ignore
    taste: Optional[Literal[*tastes]] = Field(description="What is the taste of the product", examples=tastes) # type: ignore


class Information(BaseModel):
    """Information to extract from the purchase order"""
    products: List[Product] = Field(description="list of products of the purchase order")


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.
    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    tool_calls = []
    for tool_call in example["tool_calls"]:
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "args": tool_call.dict(),
                # The name of the function right now corresponds
                # to the name of the pydantic model
                # This is implicit in the API right now,
                # and will be improved over time.
                "name": tool_call.__class__.__name__,
            },
        )
    messages.append(AIMessage(content="", tool_calls=tool_calls))
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(tool_calls)
    for output, tool_call in zip(tool_outputs, tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))

    return messages

examples = [
    (
        "Quiero jamon iberico",
        Information(products=[Product(name_by_client="jamon iberico",
                                      amount="1",
                                      type="jamon",
                                      quality="iberico",
                                      weight=None,
                                      format=None,
                                      taste=None)]),
    ),
    (
        "Quiero loncheado de salchichon",
        Information(products=[Product(name_by_client="loncheado de salchichon", 
                                    amount="1",
                                    type="loncheado salchichón",
                                    quality=None,
                                    weight=None,
                                    format=None,
                                    taste=None)]),
    ),
   (
        "Quiero loncheado de jamon y dos de miel",
        Information(products=[Product(name_by_client="loncheado de jamon", 
                                    amount="1",
                                    type="loncheado jamon",
                                    quality=None,
                                    weight=None,
                                    format=None,
                                    taste=None),
                                Product(name_by_client="miel",
                                        amount="2",
                                        type="miel",
                                        quality=None,
                                        weight=None,
                                        format="tarro",
                                        taste=None)
                                ]
                        ),
    ), 
]

def get_examples():

    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )

    return messages

