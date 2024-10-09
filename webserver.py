#
# Use  uvicorn webserver:app --host 0.0.0.0 --port 9090 to run the server
#
# Use  curl -X POST "http://localhost:9090/extract" -H "Content-Type: application/json" -d '{"text": "Quiero jamon iberico"}'
# to test it
#
from fastapi import FastAPI
from pydantic import BaseModel
from extraction_agent import execute_extractor  # Import your function

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/extract")
def extract_text(data: InputText):
    result = execute_extractor(data.text)
    return result

