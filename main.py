from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model, generate_with_vLLM_model
from prompt import rag_prompt, eval_prompt
from generator import retriever
import numpy as np
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
from verify import initialize_value_model, probability_subanswer_question, probability_subquestion_question
import torch
import time
from evaluate import run_evaluation
initialize_value_model()


evaluator = GPQAEvaluator()
tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
generator = Generator(cfg, tokenizer, model, evaluator)
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
split = 2
df = ds['train'].select(range(split)).to_pandas()
# 변수설정
reranker = True
rag_only_one = False
critic = False
output_list = []
input_list = []
if __name__ == "__main__":
  # retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",1)
# for i in range(len(ds['train']['Question'])):
  with torch.no_grad():
    for i in range(split):
      question = ds['train']['Question'][i]
      input_list.append(question)
      value_list = []
      final_questions = []
      subquestions_retrieval=[]
      best_subquestion = None  # 또는 ""로 초기화

      rag_score_dict = defaultdict()
    #-------------------subquestion------------------
      subquestions = generator.generate_subquestions(question,'')
      for i in subquestions:
        value = probability_subquestion_question(question, subquestions) # probablistic score
        value_list.append(value)
      top_indices = np.argsort(value_list)[::-1][:3]
      top3_subquestions = [subquestions[i] for i in top_indices] #top3 subquestions
      last_indices = np.argsort(value_list)[::-1][3:]
      bad_subquestions = [subquestions[i] for i in last_indices]
    #--------------------retrieval-----------------
      for subquestion in top3_subquestions:
        best_score = float('-inf')
        if reranker: #reranker
          retrieved_documents = retriever.search_document_demo(subquestion, 3)
          for retrieved_document in retrieved_documents:
            # score = llm_proposal(eval_prompt.format(rag_prompt.format(retrieved_document,subquestion)))
            score = generate_with_vLLM_model(model,eval_prompt.format(content=rag_prompt.format(context=retrieved_document['text'],question=subquestion)))
            print(f'score: {score}')
            if isinstance(score, int) and score > best_score:
              best_score = score
              best_subquestion = subquestion
              best_subquestion_retrieval = retrieved_document
          if best_subquestion:
              subquestions_retrieval.append(rag_prompt.format(context=best_subquestion_retrieval,question=best_subquestion),best_score)
          else:
              # subquestions_retrieval.append(rag_prompt.format(context=subquestion,question=retrieved_documents[0]['text']),'score가 점수아님')
              subquestions_retrieval.append((
      rag_prompt.format(context=subquestion, question=retrieved_documents[0]['text']),
      'score가 점수아님'
  ))
        else:
          pass

    #------------------subanswer------------------------
      for subquestion,score in subquestions_retrieval:
        io_output_list, subquestion_list, self_consistency_subanswer_list, value_list = generator.subanswer('',subquestion)
        # probability_subanswer_question(ori_query, answer, ans_weight=0.75): 
    #-------------------final_output--------------------
      output, only_answer = final_output(user_question, final_questions,self_consistency_subanswer_list)
      torch.cuda.empty_cache()
      output_list.append(only_answer)
    run_evaluate(df,input_list,output_list, output_dir='/workspace/output')
# def run_evaluation(df, input_list, output_list,start_index=0, dataset_name='gpqa', output_dir='/content/output', split=1, apply_backoff=False):





  
