from typing import Dict, List


class PromptGenerator:
    @staticmethod
    def create_prompt(documents: List[Dict[str, str]], query: str) -> str:
        if not documents:
            return f"Não foi possível encontrar informações relevantes para a pergunta: {query}"

        template = """
        Considere o seguinte documento ao responder à pergunta. Não tente inventar uma resposta:
        {documents}

        Pergunta do usuário: {query}

        Resposta:
        """
        return template.format(
            documents="\n".join([f"P: {doc['pergunta']}\nR: {
                                doc['resposta']}" for doc in documents]),
            query=query

        )
