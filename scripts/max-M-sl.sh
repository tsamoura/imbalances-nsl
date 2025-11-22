###### MAX-M, M=3, IR = 0,5,15,50,70 SL loss  

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=5-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 5 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=15-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 15 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=50-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 50 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-IR=70-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type expr --imb_ratio 70 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-MMAX-SCALLOP-M=3-topk=1-NS=3000-EXPR-ORIGINAL-EPOCHS=100-seed=2 \
--num-class 10 --dataset mmax --M 3 --top-k 1 --imb_type original --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2
