# get OSM tree
python ADAPTIVE_with_BASEMODEL_train.py --part_num=0 --save_dir='experimentX' --iter=5
python ADAPTIVE_with_BASEMODEL_train.py --part_num=1 --save_dir='experimentX' --iter=5
python ADAPTIVE_with_BASEMODEL_train.py --part_num=2 --save_dir='experimentX' --iter=5
python ADAPTIVE_with_BASEMODEL_train.py --part_num=3 --save_dir='experimentX' --iter=5
python ADAPTIVE_with_BASEMODEL_train.py --part_num=4 --save_dir='experimentX' --iter=5
python ADAPTIVE_with_BASEMODEL_train.py --part_num=5 --save_dir='experimentX' --iter=5

# get performance
python ADAPTIVE_with_BASEMODEL_get_info.py --n_parts=6 --save_dir='experimentX' --iter=5
