import argparse
import logging
from dotenv import load_dotenv
import subprocess, sys

from DB.prompts.PromptManager import PromptManager
from DB.LLM_storage.ResponseManager import ResponseManager
from DB import db_conn
from LLM.ResponseGenerator import ResponseGenerator
from metrics.traditional.scorer import metric_scorer
from LLM.clients import azure_client
from LLM.judge.judge import LLMAsJudge
from LLM.judge.helper import save_judge_scores

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename="app.log"
)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        subprocess.run(["streamlit", "run", "dashboard.py"])
        logger.info("Streamlit visualisations available...")
    else:
        logger.info("Establishing connection to DB")

        conn = db_conn.get_connection()
        if conn is None:
            logger.error("Failed to connect to DB")
            exit(1)
        
        logger.info("Connection to DB established, loading prompts...")

        prompt_manager = PromptManager(conn)
        llm_manager = ResponseManager(conn)

        run_type = int(input("Evaluate by...\n" \
                            "1 - Prompt ID\n" \
                            "2 - Task Type\n"
                            "3 - Multiple IDs\n"
                            ">>> "))

        prompts = []
        try:
            if run_type == 1:
                id = input("Enter Prompt ID >>> ")

                # Dealing with single prompt as list as makes it easier to 
                # handle in rest of main
                prompts = [prompt_manager.load_prompt_by_id(id)]
            elif run_type == 2:
                task_type = input("Enter task type >>> ")
                prompts = prompt_manager.load_prompts_by_task(task_type)

            elif run_type == 3:
                num_evals = int(input("Enter num of IDs you want to evaluate: "))

                ids = [int(input(f"Enter Prompt ID {i+1}: ")) for i in range(num_evals)]
                prompts = prompt_manager.load_prompts_by_ids(ids)
                    


        except Exception as e:
            logger.error(f"Integer value of 1-3 expected got {run_type} - {e}")
            exit(1)


        if not prompts:
            if run_type == 1:
                logger.error(f"No prompts found for Prompt {id}")
            elif run_type == 2:
                logger.error(f"No prompts found for Task Type {task_type}")
            else:
                logger.error(f"No prompts found for IDs {ids}")
            conn.close()
            exit(1)

        few_shot_example = None
        if prompts[0].task_type.upper() == 'SUMMARISATION' and len(prompts) > 1:
            few_shot_example = prompts.pop(0)
            logger.info(f"Using prompt {few_shot_example.id} as few-shot example for summarisation")

        clients = [azure_client.AzureClient()]
        judge = LLMAsJudge()

        logger.info("Checking if batch possible...")

        # HARDCODED VAL - Address
        if len(prompts) < 10:
            logger.info("Not batching, not enough prompts")
            logger.info("Generating responses from LLMs - No Batching")

            for prompt in prompts:
                generator = ResponseGenerator(prompt, clients, few_shot_example)
                reference = prompt.highlights if prompt.task_type.upper() == 'SUMMARISATION' else prompt.reference_output
                for gen_data in generator.generations:
                    if gen_data['llm_response'] is None:
                        logger.warning(f"Skipping prompt {prompt.id} - no response returned (content filter or API error)")
                        continue
                    llm_manager.save_generations(gen_data)
                    metric_scorer(gen_data['llm_response'],
                                reference, conn, prompt.id,
                                gen_data['response_id'],
                                prompt.task_type,
                                batch_id=None)
                    judge_result = judge.evaluate(prompt, gen_data['llm_response'])
                    save_judge_scores(judge_result, gen_data['response_id'], prompt.id, conn, prompt.task_type)

            conn.close()

        else:
            batches = prompt_manager.batch_prompts(prompts, batch_size=10)

            logger.info(f"Created {len(batches)} from {len(prompts)}")

            for batch in batches:
                logger.info(f"Processing batch - ID: {batch.batch_id} with {batch.size()}")

                for prompt in batch.prompts:
                    generator = ResponseGenerator(prompt, clients, few_shot_example)
                    reference = prompt.highlights if prompt.task_type.upper() == 'SUMMARISATION' else prompt.reference_output
                    for gen_data in generator.generations:
                        if gen_data['llm_response'] is None:
                            logger.warning(f"Skipping prompt {prompt.id} - no response returned (content filter or API error)")
                            continue
                        llm_manager.save_generations(gen_data)
                        metric_scorer(gen_data['llm_response'],
                                    reference, conn, prompt.id,
                                    gen_data['response_id'],
                                    prompt.task_type,
                                    batch_id=batch.batch_id)
                        judge_result = judge.evaluate(prompt, gen_data['llm_response'])
                        save_judge_scores(judge_result, gen_data['response_id'], prompt.id, conn, prompt.task_type, batch_id=batch.batch_id)
            conn.close()

    
