FROM python:3.11-slim

# 作業ディレクトリを変更
WORKDIR /workspace

RUN apt-get update -y -q && \
    apt-get install -q -qq -y git

# ローカルPCのファイルをコンテナのカレントディレクトリにコピー
COPY ./requirements.txt ${pwd}

# pipのアップデート
RUN pip install --upgrade pip

# pythonパッケージをインストール
RUN pip install -r requirements.txt

# # コンテナ起動時に実行するコマンドを指定
# CMD ["/bin/bash"]