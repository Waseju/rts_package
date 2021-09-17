FROM python:3.8.1-alpine

# A few Utilities to able to install C based libraries such as numpy
RUN apk update
RUN apk add make automake gcc g++ git libc-dev

RUN pip install --upgrade pip setuptools wheel
RUN pip install rts_package==1.0.3

CMD rts_package
