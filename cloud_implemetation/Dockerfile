FROM python:3.13.5-slim
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y libxcb1 && apt-get install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0
RUN pip install -r requirements.txt
EXPOSE 8080
CMD streamlit run app.py --server.port 8080