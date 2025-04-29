from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model
from prompt import rag_prompt, eval_prompt
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
from easydict import EasyDict as edict
import math

cfg = edict()

cfg.note = "debug"

cfg.api = "vllm"
cfg.allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt-4o"]

cfg.seed = 42
cfg.verbose = False
cfg.tensor_parallel_size = 1
cfg.half_precision = False
# WandB settings
cfg.wandb_mode = "disabled"  # options: ["disabled", "online"]
# LLM settings
# cfg.model_ckpt = "google/gemma-3-1b-it"  # <-- 반드시 수동으로 설정해야 함
cfg.model_ckpt = "meta-llama/Llama-3.2-1B-Instruct"
cfg.model_parallel = False
cfg.half_precision = False
cfg.max_tokens = 1024
cfg.temperature = 0.4
cfg.top_k = 40
cfg.top_p = 0.9
cfg.num_beams = 3
# cfg.repetition_penalty = 1.1
cfg.max_num_worker = 3
cfg.test_batch_size = 1
cfg.tensor_parallel_size = 1

# prompt settings
cfg.prompts_root = "prompts"

# dataset settings
cfg.data_root = "data"
cfg.allowed_dataset_names = [
    "FMT", "GPQA", "WICE", "CWEBQA", "MATH", "GSM8K", "GSM8KHARD",
    "STG", "SVAMP", "MULTIARITH", "ScienceQA", "SciKEval", "CFA"
]
cfg.dataset_name = "GPQA"  # <-- 반드시 실제 사용 전에 바꿔야 함
cfg.test_json_filename = "test_all"
cfg.start_idx = 0
cfg.end_idx = math.inf
# outputs settings
cfg.run_outputs_root = "run_outputs"
cfg.eval_outputs_root = "eval_outputs"

cfg.temperature = 0.8
cfg.top_p = 0.95
cfg.top_k = 40
cfg.repetition_penalty = 1.1
cfg.n = 1
cfg.max_tokens = 256
cfg.logprobs = 1
cfg.stop = []
cfg.disable_rag = True
cfg.num_subquestions = 3
cfg.num_votes = 3
cfg.max_tokens = 256
cfg.enable_potential_score = True

cfg.mcts_num_last_votes = 3
# generator arg
        # if not args.disable_rag:
        #     self.retriever = Retriever()
        #     self.retriever.regist_io_system(self.io)

        # self.num_subquestions = args.num_subquestions
        # self.num_a1_steps = args.num_a1_steps
        # self.num_votes = args.num_votes
        # self.max_tokens = args.max_tokens
        # self.enable_potential_score = args.enable_potential_score

        # self.mcts_num_last_votes = args.mcts_num_last_votes

        # with open(args.decompose_template_path, "r") as f: