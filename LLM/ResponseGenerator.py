import os
from dotenv import load_dotenv
import logging
from uuid import uuid4

load_dotenv()

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, prompt, clients, few_shot_example=None):
        self.prompt = prompt
        self.generations = [client.generate(prompt, few_shot_example) for client in clients]