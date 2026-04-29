# Final-Year-Project

This project takes a look at traditional metrics initally used for natural language processing techniques for evaluating LLMs such as BLEU and ROUGE with the aim of proving why these traditional metrics do not meet the requirements given the robust demands placed on LLMs today. I aim to create a standardised approach taking influence from efforts from up to date literature of the field today. The project will be broken into two phases...

# Phase 1
In phase 1, the aim is to use traditional metrics for evaluating LLMs as mentioned above to generate a similarity score based on the LLM accuracy given a prompt and standard reference answer. This models how these metrics are used in the field of LLM evaluation, the results gathered here will provide a baseline which will stand as a point of reference in proving why these metrics are not good enough given how much LLMs have evolved since the introduction of these metrics.

# Phase 2

This part of the project will introduce a new framework for accurately depicting LLM performance, I will be using methodologies such as LLM-As-Judge and Reasoning Analysis to better understand the quality of LLM output. We can examine how well the LLM reasons and not just rely on a semantic similarity score which doesn't capture real insights behind the output, allowing us to not just assume but affirm that the LLM is performing positively. The judge will be used to quantify the quality of the LLMs response also, which is not something BLEU or ROUGE provide, using certain criterion like fluency, coherency, factual accuracy and more ensuring the LLMs response is accurate and has sound logical reasoning behind to back it up.
