FROM python:3.10.6

COPY . .

RUN pip install -r requirements.txt
CMD ["python", "./app/server.py"]
