# train.py #
python train.py --config config/bert4rec.yaml

nohup python train.py --config config/bert4rec.yaml > train.log 2>&1 &

# inference.py #
python inference.py --checkpoint saved/CDAE-best.pth --topk 100

