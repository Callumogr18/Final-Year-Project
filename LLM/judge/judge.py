import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from LLM.judge.pydantic_models import JudgeEvaluation

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. "
    "You will be given an input, a context, and a model response. "
    "Answer each question with YES or NO only. No explanation."
)


METRIC_QUESTIONS = {
    # Metric
    "Hallucination": [
        # Sub questions for every metric
        "Is every factual claim in the response supported by the provided context?",
        "Does the response avoid introducing names, dates, or figures not present in the context?",
        "Does the response avoid contradicting any information in the context?",
        "Does the response stay within the scope of the provided context?",
    ],
    "Fluency": [
        "Is the response free from grammatical errors?",
        "Does the response read naturally without awkward phrasing?",
        "Are all sentences complete and well-formed?",
        "Is the language clear and appropriate for the task?",
    ],
    "Consistency": [
        "Is the response internally consistent with no self-contradictions?",
        "Does the response maintain a consistent viewpoint or stance throughout?",
        "Does the response avoid repeating the same claim with conflicting details?",
        "Is the response consistent with the information provided in the context?",
    ],
    "Reasoning": [
        "Does the response directly address the question or task?",
        "Are the conclusions in the response logically supported by the reasoning given?",
        "Is the reasoning free from obvious logical fallacies?",
        "Does the response follow a clear logical thought process?",
    ],
    "Coherence": [
        "Does the response have a clear and logical structure?",
        "Do ideas flow logically from one to the next?",
        "Is the response focused and on-topic throughout?",
        "Does the response avoid unnecessary tangents or contradictions?",
    ],
    "Factual Accuracy": [
        "Are all named entities (people, places, organisations) accurately represented?",
        "Are numerical values such as dates, figures, and statistics correct relative to the context?",
        "Does the response avoid making claims beyond what the context supports?",
        "Are cause-and-effect relationships accurately stated?",
    ],
}

class LLMAsJudge:
    def __init__(self):
        self.client = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_JUDGE_MODEL"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_V"),
            temperature=0,
            max_retries=2,
        )

    def build_message(self, prompt, llm_response):
        context = (
            "\n".join(prompt.contexts) if isinstance(prompt.contexts, list)
            else prompt.contexts or prompt.article or ""
        )

        # For prompts with no context (e.g. TruthfulQA), fall back to the
        # reference answer so the judge has a ground truth to evaluate against.
        if not context and prompt.reference_output:
            context_block = f"Reference Answer:\n{prompt.reference_output}"
        else:
            context_block = f"Context:\n{context}"

        questions_block = ""
        for metric, questions in METRIC_QUESTIONS.items():
            questions_block += f"\n{metric}:\n"
            for i, q in enumerate(questions, 1):
                questions_block += f"  {i}. {q}\n"

        human_content = (
            f"Input:\n{prompt.input_text}\n\n"
            f"{context_block}\n\n"
            f"Response:\n{llm_response}\n\n"
            f"For each metric below, answer every sub-question YES or NO.\n"
            f"{questions_block}"
        )

        return [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def evaluate(self, prompt, llm_response):
        messages = self.build_message(prompt, llm_response)
        structured_client = self.client.with_structured_output(JudgeEvaluation)
        result: JudgeEvaluation = structured_client.invoke(messages)

        for metric in result.metrics:
            logger.info(metric.summary())

        return result
