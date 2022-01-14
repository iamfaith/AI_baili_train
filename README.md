
find `pwd`/train -name \*.png > train.list
find `pwd`/test -name \*.png > test.list


find `pwd`/images -name \*.png > train.list



python export.py --weights /home/faith/AI_baili_train/best5000.pt --include onnx --img 384 768
