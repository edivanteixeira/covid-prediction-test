
FROM python:3.8

WORKDIR /main
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py ./

CMD [ "streamlit", "run",  "./main.py" ]