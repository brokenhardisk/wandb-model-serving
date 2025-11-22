# TODO: Dockerfile for Frontend Application
FROM python:3.9-slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

EXPOSE ${STREAMLIT_SERVER_PORT}

CMD ["streamlit", "run", "streamlit_app.py"]