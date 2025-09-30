import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part, ContentEmbedding, File, ContentDict


class EmbeddingEngineWrapper:
    def __init__(self, model_name="text-embedding-004"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

    def embed(self, messages:  list[Content | list[File | Part | None | str] | File | Part | None | str]
                               | Content
                               | list[File | Part | None | str]
                               | File
                               | Part
                               | None
                               | str
                               | list[Content | list[File | Part | None | str] | File | Part | None | str | ContentDict]
                               | ContentDict) -> list[ContentEmbedding] | None:
        response = self.client.models.embed_content(model=self.model_name, contents=messages)
        return response.embeddings


def test():
    wrapper = EmbeddingEngineWrapper()
    prompt = "Explain quantum entanglement in simple terms."
    print(wrapper.embed(prompt))
    print()

if __name__ == "__main__":
    load_dotenv()
    test()
