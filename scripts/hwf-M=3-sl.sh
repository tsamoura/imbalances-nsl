###### HWF-M, M=3, IR = 5,15,50,70,original, seed=1 SL 

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=5-EPOCHS=100-seed=1 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 5 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 1 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=15-EPOCHS=100-seed=1 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 15 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 1 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=50-EPOCHS=100-seed=1 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 50 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 1 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=70-EPOCHS=100-seed=1 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 70 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 1 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-ORIGINAL-EPOCHS=100-seed=1 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type original --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 1 


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=5-EPOCHS=100-seed=2 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 5 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=15-EPOCHS=100-seed=2 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 15 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=50-EPOCHS=100-seed=2 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 50 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=70-EPOCHS=100-seed=2 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 70 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-ORIGINAL-EPOCHS=100-seed=2 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type original --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 2 


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=5-EPOCHS=100-seed=3 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 5 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 3 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=15-EPOCHS=100-seed=3 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 15 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 3 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=50-EPOCHS=100-seed=3 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 50 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 3 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-EXP-IR=70-EPOCHS=100-seed=3 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type exp --imb_ratio 70 --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 3 --imb_test

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_scallop.py --exp-dir experiment/MIPLL-HWF-SCALLOP-M=3-topk=1-NS=1000-ORIGINAL-EPOCHS=100-seed=3 \
--num-class 13 --dataset hwf --M 3 --top-k 1 --imb_type original --size_partial_dataset 1000 --batch-size 64 --save_ckpt --epochs 100 --seed 3 
