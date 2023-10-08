FROM python:3.10
WORKDIR /app
COPY cdata.csv .
COPY code.py .
RUN pip install scikit-learn pandas
CMD python code.py