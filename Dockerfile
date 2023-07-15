FROM python:3.8-slim-buster

RUN pip install transformers accelerate && \
    pip install vncorenlp && \
    pip install Flask

# install open jdk for java
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f

# install maven for java
RUN apt install wget -y && \
    wget https://mirrors.estointernet.in/apache/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && \
    tar -xvf apache-maven-3.6.3-bin.tar.gz && \
    mv apache-maven-3.6.3 /opt/ && \
    M2_HOME='/opt/apache-maven-3.6.3' && \
    PATH="$M2_HOME/bin:$PATH" && \
    export PATH

# install vncorenlp packet
RUN mkdir src && cd src && \
    mkdir -p vncorenlp/models/wordsegmenter && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab && \
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr && \
    mv VnCoreNLP-1.1.1.jar vncorenlp/ && \
    mv vi-vocab vncorenlp/models/wordsegmenter/ && \
    mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

RUN rm -r apache-maven-3.6.3-bin.tar.gz

RUN apt install libpq-dev python-dev -y && \
    apt-get -y install gcc && \
    python3 install.py && \
    pip install -U flask-cors && \
    pip install psycopg2

WORKDIR src

RUN mkdir transformer_model
COPY transformer_model ./transformer_model

COPY install.py constants.py server.py vietnamese-stopwords-dash.txt ./

EXPOSE 8081

CMD ["python3", "server.py"]