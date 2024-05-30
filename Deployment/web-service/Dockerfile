FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv  install --system --deploy

COPY ["predict.py", "linear_regression_model.bin", "./"]

EXPOSE 9696 

ENTRYPOINT ["waitress-serve", "--port=9696", "predict:app"]


