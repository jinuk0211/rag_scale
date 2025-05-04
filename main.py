from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model
from prompt import rag_prompt, eval_prompt
from generator import retriever
import numpy as np
from cfg import cfg
evaluator = GPQAEvaluator()
tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
generator = Generator(cfg, tokenizer, model, evaluator)

for i in range(len(ds['train']['Question'])):
  question = ds['train']['Question'][i]
  value_list = []
  final_questions = []
  subquestions_retrieval=[]
  best_score = float('-inf')
  rag_score_dict = defaultdict()
#-------------------subquestion------------------
  subquestions = generator.generate_subquestions(question,'')
  for i in subquestions:
    value = probability_subquestion_question(subquestions) # probablistic score
    value_list.append(value)
  top_indices = np.argsort(value_list)[::-1][:3]
  top3_subquestions = [subquestions[i] for i in top_indices] #top3 subquestions
  last_indices = np.argsort(value_list)[::-1][3:5]
  bad_subquestions = [subquestions[i] for i in last_indices]
#--------------------retrieval-----------------
  for subquestion in top3_subquestions:
    if reranker: #reranker
      retrieved_documents = retriever.search_document_demo(subquestion, 3)
      for retrieved_document in retrieved_documents:
        score = llm_proposal(eval_prompt.format(rag_prompt.format(retrieved_document,subquestion)))
        if score > best_score:
          best_score = score
          best_subquestion = subquestion
          best_subquestion_retrieval = retrieved_document
      subquestions_retrieval.append(rag_prompt.format(best_subquestion_retrieval,best_subquestion),'best 점수만 뽑은 것')

    else:
      pass
      #subquestions의 retrieval을 평가
    if rag_only_one:#1개만 retrieve하고 subquestion들의 rag 프롬프트 비교
      top_doc = retriever.search_document_demo(i, 1)[0]
      subquestion_retrieval_score_prompt = eval_prompt.format(top_doc,subquestion)
      score = llm_proposal(subquestion_retrieval_score_prompt)    # llm score
      subquestions_retrieval.append((rag_prompt.format(top_doc,subquestion), score))
      subquestions_retrieval = sorted(subquestions_retrieval, key=lambda x: x[1], reverse=True)
    else:
      pass

#-------------------critic 별로 안중요함-----------------
    if critic and score < 9: #수정 할시
      subquestions = generator.rephrased_question(question,'',bad_subquestions)
      retrieved_context = generator.retrieve(question,'',subquestions)
      top_doc = retriever.search_document_demo(query_1, 1)[0]
      score = llm_proposal(retrieved_context,subquestions)
#------------------subanswer------------------------
  for subquestion,score in subquestions_retrieval:
    io_output_list, subquestion_list, self_consistency_subanswer_list, value_list = generator.subanswer('',subquestion)
    # probability_subanswer_question(ori_query, query, answer, ans_weight=0.75):
#-------------------final_output--------------------
  final_output, only_answer = final_output(user_question, final_questions,self_consistency_subanswer_list)
evaluator.evaluate(, ds['train']['Correct Answer'][0])


# !pip install easydict


if __name__ == "__main__":
  retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",1)