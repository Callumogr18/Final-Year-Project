from typing import List
from uuid import UUID
import logging

logger = logging.getLogger(__name__)

from .Prompt import Prompt
from .PromptBatch import PromptBatcher, PromptBatch

class PromptManager:
    def __init__(self, db_session):
        self.db = db_session
        self.cursor = db_session.cursor()
        self._batcher = PromptBatcher()

        logger.info(f'PromptManager initialised')

    def load_prompts_by_task(self, task_type: str) -> List[Prompt]:
        valid_tasks = ['QA', 'REASONING', 'SUMMARISATION']

        if task_type.upper() not in valid_tasks:
            logger.error(f"Invalid task type: {task_type}")
            return []
        
        logger.info(f"Getting {task_type} prompts from DB...")

        self.cursor.execute(
            '''SELECT "id", "task_type", "question", "ground_truths", 
            "answer", "contexts", "article", "highlights" 
            FROM public.prompts WHERE "task_type" = %s''',
            (task_type,)
        )

        rows = self.cursor.fetchall()

        logger.info(f"Gathered {len(rows)} from DB")

        prompts = []
        for row in rows:
            prompt = Prompt(
                id=row[0],
                task_type=row[1],
                input_text=row[2],
                reference_output=row[3],
                answer=row[4],
                contexts=row[5],
                article=row[6],
                highlights=row[7]
            )
            prompts.append(prompt)

        return prompts


    def load_prompt_by_id(self, id: int) -> Prompt:
        logger.info(f"Fetching prompt {id} from DB...")

        self.cursor.execute(
            '''SELECT "id", "task_type", "question", "ground_truths",
            "answer", "contexts", "article", "highlights"
            FROM public.prompts WHERE "id" = %s''',
            (id,)
        )

        row = self.cursor.fetchone()
        if row is None:
            logger.error(f"No Prompt {id} in DB")
            return None

        return Prompt(
            id=row[0],
            task_type=row[1],
            input_text=row[2],
            reference_output=row[3],
            answer=row[4],
            contexts=row[5],
            article=row[6],
            highlights=row[7]
        )

    def load_prompts_by_ids(self, ids: List[int]) -> List[Prompt]:
        logger.info(f"Fetching {len(ids)} prompts from DB...")

        self.cursor.execute(
            '''SELECT "id", "task_type", "question", "ground_truths",
            "answer", "contexts", "article", "highlights"
            FROM public.prompts WHERE "id" = ANY(%s)''',
            (ids,)
        )

        rows = self.cursor.fetchall()
        logger.info(f"Gathered {len(rows)} prompts from DB")

        return [
            Prompt(
                id=row[0],
                task_type=row[1],
                input_text=row[2],
                reference_output=row[3],
                answer=row[4],
                contexts=row[5],
                article=row[6],
                highlights=row[7]
            )
            for row in rows
        ]
    

    def batch_prompts(self, prompts: List[Prompt], batch_size: int = 10) -> List[PromptBatch]:
        self._batcher._default_batch_size = batch_size
        return self._batcher.create_batches(prompts)