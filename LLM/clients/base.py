from abc import ABC, abstractmethod

SYSTEM_PROMPTS = {
    'QA': 'You are a question answering assistant. Answer using only the shortest possible phrase or span from the provided context. Do not write full sentences or add any explanation.',
    'SUMMARISATION': 'You are a news summarisation assistant. Write a piece summarising the article, ensure that length is appropriate for a summary. Each sentence must use the specific names, places, and facts from the article directly. Do not paraphrase or generalise. Do not add information beyond what is provided.'
}

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass