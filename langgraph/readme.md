pyenv install 3.11.10
pyenv local 3.11.10

#this will create a .python-version file


python -m venv env

source env/bin/activate
pip install --upgrade pip



pip install -U langchain
pip install -qU "langchain[anthropic]"