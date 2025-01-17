# Use a imagem oficial do Python 3.13
FROM python:3.13-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie os arquivos de dependências para o container
COPY requirements.txt /app/

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante dos arquivos para o container
COPY . /app
