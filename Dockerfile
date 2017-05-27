FROM python:3
WORKDIR /ParlAI
ADD . /ParlAI/
RUN pip install -r requirements.txt
RUN python setup.py develop
CMD cd tests && ./run_tests_short.sh
