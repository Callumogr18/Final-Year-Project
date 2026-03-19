from pydantic import BaseModel


class SubQuestionAnswer(BaseModel):
    """
    A single yes/no question and its boolean answer for a specific evaluation criteria

    Used as a sub-component of a MetricEvaluation, where each metric is broken down
    into a set of questions answered by the judge

    Attributes:
        question: The evaluation question asked of the judge
        answer: The judge's boolean response (True = yes, False = no).
    """
    question: str
    answer: bool


class MetricEvaluation(BaseModel):
    """
    The LLM judge's evaluation of a single metric (e.g. Hallucination, Fluency)

    Aggregates a list of yes/no sub-questions for the metric and provides helpers
    to compute a normalised score and a human-readable summary

    Attributes:
        metric: Name of the evaluation metric (e.g. "Hallucination")
        answers: List of sub-questions and their boolean answers from the judge
    """
    metric: str
    answers: list[SubQuestionAnswer]

    def yes_count(self) -> int:
        return sum(1 for a in self.answers if a.answer)

    def score(self) -> float:
        return self.yes_count() / len(self.answers) if self.answers else 0.0

    def summary(self) -> str:
        lines = [f"{self.metric}: {self.yes_count()}/{len(self.answers)}"]
        for a in self.answers:
            lines.append(f"  {'YES' if a.answer else 'NO '} — {a.question}")
        return "\n".join(lines)


class JudgeEvaluation(BaseModel):
    """
    The complete structured output returned by the judge for a single response

    Wraps all per-metric evaluations into one top-level object, representing the
    judge's full assessment across every evaluation criterion

    Attributes:
        metrics: List of per-metric evaluations produced by the judge
    """
    metrics: list[MetricEvaluation]
