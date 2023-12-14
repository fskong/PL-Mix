#!/bin/bash

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s book -t dvd --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s book -t electronics --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s book -t kitchen --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s dvd -t book --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s dvd -t electronics --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s dvd -t kitchen --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s electronics -t book --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s electronics -t dvd --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s electronics -t kitchen --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s kitchen -t book --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s kitchen -t dvd --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1

python -u src/main.py ADV --model_name ../../roberta-base --dataset Benchmark --data_dir . -s kitchen -t electronics --epochs 10 \
						--seed $1 --train_batch_size 8 --test_batch_size 32 -i 100 -p 50 --lr 1e-5 --domain-lr 5e-5 --prompt-k $2 --mix_alpha_c 0.1 --mix_alpha_d 1
















