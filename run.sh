pip install -r requirements.txt
cd datasets/amazon-book/
unzip kg_final.txt.zip
cd ..
cd ..
ls
python main.py --n_epoch 100 --use_pretrain 0
