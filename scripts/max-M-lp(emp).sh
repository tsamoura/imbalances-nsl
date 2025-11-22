###### M-MAX, M=3, IR = 5,15,50,70,0, seed=2 LP(EMP) 
###### --model takes as input a model that has been previously trained with SL (on the same dataset). This parameter is optional.

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-ILP-M=3-topk=1-NS=3000-EXPR-IR=5-NAIVE-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 5 --size_naive_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 \
--ilp-training --continuous-relaxation --estimated-label-ratio naive --model "MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=5-EPOCHS=100-seed=2/checkpoint.pth.tar"

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-ILP-M=3-topk=1-NS=3000-EXPR-IR=15-NAIVE-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 15 --size_naive_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 \
--ilp-training --continuous-relaxation --estimated-label-ratio naive --model "MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=15-EPOCHS=100-seed=2/checkpoint.pth.tar"


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-ILP-M=3-topk=1-NS=3000-EXPR-IR=50-NAIVE-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 50 --size_naive_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 \
--ilp-training --continuous-relaxation --estimated-label-ratio naive --model "MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=50-EPOCHS=100-seed=2/checkpoint.pth.tar"


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-ILP-M=3-topk=1-NS=3000-EXPR-IR=70-NAIVE-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 70 --size_naive_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 \
--ilp-training --continuous-relaxation --estimated-label-ratio naive --model "MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=70-EPOCHS=100-seed=2/checkpoint.pth.tar"


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-ILP-M=3-topk=1-NS=3000-EXPR-ORIGINAL-NAIVE-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type original --size_naive_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 \
--ilp-training --continuous-relaxation --estimated-label-ratio naive --model "MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-ORIGINAL-EPOCHS=100-seed=2/checkpoint.pth.tar"
