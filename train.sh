source /home/shiqisun/Env_space/py_19_env/bin/activate
pip install addict -i https://pypi.tuna.tsinghua.edu.cn/simple

CONFIG=$1
GPUS=$2
is_distributed=$3

echo config: "$CONFIG"
echo gpu_num: "$GPUS"

python -m torch.distributed.launch --nproc_per_node=$GPUS tools/trainval/train.py --config $CONFIG 
# python3 tools/train.py $CONFIG