# Projeto de Assistente Inteligente

Este projeto é um assistente inteligente que utiliza uma coleção de documentos armazenados em um arquivo JSON para responder a perguntas com base nas respostas presentes nos documentos. O sistema utiliza embeddings para gerar representações vetoriais das respostas e armazená-las em uma instância do Qdrant. O modelo Llama3 é utilizado para gerar as respostas finais.

### Requisitos

Antes de rodar o projeto, é necessário garantir que você tenha as seguintes ferramentas instaladas:

- **Docker**: Utilizado para o gerenciamento de containers e serviços.
- **Python 3.13** ou superior: Versão do Python utilizada no projeto.
- **Ollama** com o modelo `llama3`: Serviço responsável por gerar as respostas baseadas nos prompts fornecidos.

### Como Rodar o Projeto

1. Clonar o repositório
2. Suba o container o Qdrant que é o banco de dados vetorial com o comando:
```bash
docker compose up -d
```
3. Instalar as dependências do Python com os comandos:
```bash
python3.13 -m venv venv
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