
from pydantic import BaseModel, Field
from typing import List,Optional, Literal, Set
import pandas as pd
import math

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
    """Information about a product"""
    name_by_client: str = Field(description="name used by the customer to refer this product")
    amount: str = Field(description="number of units of this product in the purchase order. Could be 1/2 as well",examples=["1","1/2","2","3"])
    type: Literal[*categories] = Field(description="type of product", examples=categories) # type: ignore
    quality: Literal[*qualities] = Field(description="Adjective that indicates quality of the product") # type: ignore
    weight: Optional[str] = Field(description="Weight of the product", examples=weights)
    format: Literal[*formats] = Field(default = "",description="Format, which can describe both the type of cut but also the packaging", examples=formats) # type: ignore
    taste: Literal[*tastes] = Field(description="What is the taste of the product", examples=tastes) # type: ignore


class Information(BaseModel):
    """Information to extract from the purchase order"""
    products: List[Product] = Field(description="list of products of the purchase order")