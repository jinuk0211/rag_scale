pip install gdown
pip install faiss-cpu
pip install vllm transformers
git clone https://github.com/AkariAsai/self-rag.git -q
cd self-rag
pip install -r requirements.txt -q
mkdir -p /workspace/self-rag/retrieval_lm/enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro
gdown https://drive.google.com/file/d/1-24buVYsvSU4laZW9FXOQG8P6bMucro8/view?usp=sharing -O /workspace/self-rag/retrieval_lm/enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl
gdown https://drive.google.com/uc?id=1YasSXY4_mRaNkgkQEeRA6y-y0WH5diN8 -O /workspace/self-rag/retrieval_lm/enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/passages_01
pip install easydict