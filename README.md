# Installation process
1. Clone the repostory
2. Go to project folder: ```cd procesado_pedidos```
3. Create a virtual environment: ```python3.11 -m venv .```
4. Install dependencies: ```pip install -r requirements.txt```
5. Create an .env file with the OpenAI api key: ```OPENAI_API_KEY=sk-xxxxxxxxxxxxx```
6. To run from the command line, execute ```python extraction_agent.py```
7. To start the web interface, execute ```streamlit run app.py ```

# To use extractor from another python module

```python
from extraction_agent import execute_extractor

print(execute_extractor("quiero jamon iberico de bellota"))
```

# Output example

```json
[{'products': [{'name_by_client': 'jamón ibérico', 'amount': '1', 'type': 'jamon', 'quality': 'iberico', 'weight': None, 'format': None, 'taste': None}]}, {'products': [{'name_by_client': 'miel', 'amount': '2', 'type': 'miel', 'quality': None, 'weight': None, 'format': 'tarro', 'taste': None}]}]
```
