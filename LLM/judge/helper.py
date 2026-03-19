from uuid import uuid4

def scores_to_dict(result):
    """
    Flatten a JudgeEvaluation to a plain dict keyed by normalised metric name.
    """
    return {
        m.metric.lower().replace(" ", "_"): m.score()
        for m in result.metrics
    }


def save_judge_scores(results, response_id: str,
                      prompt_id: int, conn, task_type: str, batch_id=None):
    """
    Persist judge scores to the metrics table.
    """
    scores = scores_to_dict(results)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO public.judge_metrics (
            score_id, response_id, prompt_id, task_type,
            hallucination, fluency, consistency, reasoning, coherence,
            accuracy
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            str(uuid4()),
            response_id,
            prompt_id,
            task_type,
            scores.get("hallucination"),
            scores.get("fluency"),
            scores.get("consistency"),
            scores.get("reasoning"),
            scores.get("coherence"),
            scores.get("factual_accuracy"),
        ),
    )
    conn.commit()
