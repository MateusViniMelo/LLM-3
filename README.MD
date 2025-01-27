# QA-RAG com Ollama e Qdrant

Este projeto implementa uma funcionalidade de **Retrieval-Augmented Generation (RAG)** que combina o poder do **Ollama** (usando o modelo Llama 3) para geração de respostas com o **Qdrant** como banco de dados vetorial para recuperação de documentos relevantes./

### Funcionalidades

* **Carregamento de documentos** a partir de um arquivo JSON.
* **Indexação de documentos** no Qdrant usando embeddings gerados pelo modelo `all-MiniLM-L6-v2`.
* **Busca de documentos relevantes** com base em similaridade vetorial.
* **Geração de respostas contextualizadas** usando documentos recuperados.
* **Detecção de lacunas de conhecimento** : informa ao usuário quando os documentos não contêm informações relevantes.

### Requisitos

Antes de rodar o projeto, é necessário garantir que você tenha as seguintes ferramentas instaladas:

- **Docker**: Utilizado para o gerenciamento de containers e serviços.
- **Python 3.12** ou superior: Versão do Python utilizada no projeto.
- **Ollama** com o modelo `llama3`: Serviço responsável por gerar as respostas baseadas nos prompts fornecidos.

### Como Rodar o Projeto

1. Clonar o repositório
2. Suba o container o Qdrant que é o banco de dados vetorial com o comando:

```bash
docker compose up -d
```

3. Instalar as dependências do Python com os comandos:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Execute a aplicação e faça as perguntas:

```bash
python app/main.py
```

### Configurar o arquivo documentos.json

Certifique-se de ter o arquivo documentos.json no diretório  do projeto. Este arquivo deve conter uma lista de objetos, com cada objeto tendo as chaves pergunta e resposta, como mostrado abaixo:

```bash
[
  {
    "pergunta": "Qual é a capital da França?",
    "resposta": "Paris"
  },
  {
    "pergunta": "Qual é a fórmula da água?",
    "resposta": "H2O"
  }
]

```
