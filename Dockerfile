FROM python:3.8.12-slim-bullseye

# A few Utilities to able to install C based libraries such as numpy
RUN apt update

RUN pip install --upgrade pip setuptools wheel
RUN pip install rts_package==1.1.1

CMD rts_package
