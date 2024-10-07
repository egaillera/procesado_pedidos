# How to install

1. Clone the repository
2. Go to procesado_pedidos folder and create a python virtual environment: ``` python3 -m venv .```
3. Activate the venv: ``` source bin/activate ```
4. Make you sure to install the dependencies: ``` pip install -r requirements.txt ```
5. Crate an .env file with _OPENAI_API_KEY = sk-xxxxxxxxxxxx_
6. To run in command line mode, execute ```python extraction_agent.py```
7. To run the website, execute ```streamlit run app.py```
