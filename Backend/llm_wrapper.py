import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part


class LLMWrapper:
    def __init__(self, model_name="gemini-2.5-pro"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

    def generate(self, messages: Content | Part | str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=messages)
        return response.text

    def stream_generate(self, messages: Content | Part | str):
        for chunk in self.client.models.generate_content_stream(model=self.model_name, contents=messages):
            yield chunk.text


def test():
    wrapper = LLMWrapper()
    prompt = "Explain quantum entanglement in simple terms."
    print("Complete:")
    print(wrapper.generate(prompt))
    print("Streamed:")
    for piece in wrapper.stream_generate(prompt):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print()

if __name__ == "__main__":
    load_dotenv()
    test()
