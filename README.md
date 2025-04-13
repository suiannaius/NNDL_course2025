# NNDL_course2025

## 依赖库
- numpy
- matplotlib
- tqdm

## 训练
### 默认参数
`python run_training.py`
### 自定义参数
`python run_training.py --task_id 1 --hiddenlayer_1 1024 --hiddenlayer_2 512 --batch_size 128 --activation relu --weight_decay 1e-2`
## 测试
`python run_inference.py`
## 参数搜索
`python search.py`
