FROM python:3.10.6

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "Home.py"]

