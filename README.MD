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
- **Ollama** com o modelo `llama3`: Serviço responsável por gerar as respostas baseadas nos prompts fornecidos.

### Como Rodar o Projeto

#### Python e Qdrant no Docker

1. Clonar o repositório
2. Crie um arquivo **.env** na raiz do projeto e copie tudo o que está no **.env.example** para ele
3. Suba o container do projeto que tem o Qdrant e o Pyhton 3.12:

```bash
docker compose up -d
```

4. Entre no container que tem o

```bash
docker exec -it python_app bash
```

4. Instale as dependências do Projeto:

```bash
pip install -r requirements.txt
```

5. Execute a aplicação e faça a pergunta:

```
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
