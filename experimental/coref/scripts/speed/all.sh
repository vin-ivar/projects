mkdir -p results
mkdir -p texts
wget https://github.com/explosion/projects/releases/download/reddit/reddit-100k.jsonl -P texts
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -o experimental_coref python benchmark_speed.py ./texts ./results spacy ${1} True --n-texts 10000