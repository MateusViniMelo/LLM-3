from typing import List, Dict

class PromptGenerator:
    @staticmethod
    def create_prompt(documents: List[Dict[str, str]], query: str) -> str:
        template = """
        Você é um assistente que responde com base nos seguintes documentos:
        {documents}

        Pergunta do usuário: {query}

        Resposta:
        """
        return template.format(
            documents="\n".join([f"P: {doc['pergunta']}\nR: {doc['resposta']}" for doc in documents]), query=query
        )