
from generator import Generator, load_vLLM_model, generate_with_vLLM_model
from prompt import rag_prompt
from generator import retriever
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time


tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

if __name__ == "__main__":
  # retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",1)
# for i in range(len(ds['train']['Question'])):
  with torch.no_grad():
    for i in range(2):
        question = ds['train']['Question'][i]
        retrieved_document = retriever.search_document_demo(question, 1)
        response = generate_with_vLLM_model(model,rag_prompt.format(context=retrieved_document['text'],question=question))
        final_input = 'output:' + response _ "Therefore, the answer is "
        only_answer = generate_with_vLLM_model(model,final_input,max_tokens=20)

    torch.cuda.empty_cache()
    evaluator.evaluate(only_answer, ds['train']['Correct Answer'][i])



def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,)

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output




  
