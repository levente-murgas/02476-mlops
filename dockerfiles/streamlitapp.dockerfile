FROM python:3.9-slim

EXPOSE 8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]