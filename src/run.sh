#!/usr/bin/bash
#*************************train
#CUDA_VISIBLE_DEVICES=0 python train/train_keras.py --batch_size 128 --lr 0.0025 --pretrained_file /home/lixiaoyu80/models/mobilephone/mobile_res50_best.h5
#CUDA_VISIBLE_DEVICES=0 python train/train.py --lr 0.000035 --dataset allian --cuda true --save_folder /home/lixiaoyu80/models/mobilephone/ --batch_size 128 --multigpu false  --resume /home/lixiaoyu80/models/mobilephone/mobile_allian_best_c3.pth
#CUDA_VISIBLE_DEVICES=0 python train/train.py --lr 0.00035 --dataset allian --cuda true --save_folder /home/lixiaoyu80/models/mobilephone/ --batch_size 32 --multigpu false  --resume /home/lixiaoyu80/models/mobilephone/broken_allian_best_resnest50.pth

#**convert model
# python test/tr2tf.py

#********************utils
#python utils/scripts.py

#********************test 
#CUDA_DEVICES_VISIBLE=0 python test/eval.py --file-in ../data/test_online2.txt --base-dir /home/lixiaoyu80/Datas/onlinev5_label --out-file ../data/test.txt  --record-file ../data/test.csv  --modelpath /home/lixiaoyu80/models/mobile_allian_best_res50_allv1.pth --cmd-type evalue #/mobile_allian_best.pth/model_v1.h5
#CUDA_DEVICES_VISIBLE=0 python3 test/eval.py --file-in ../data/test_broken_p24.txt --base-dir /home/lixiaoyu80/Datas/p24_front --out-file ../data/broken_org_p24_result.txt  --record-file ../data/broken_org_p24_result.csv  --modelpath /home/lixiaoyu80/models/broken_phone_model_v2_many_labels_v2.pb --cmd-type evalue #/mobile_allian_best.pth/model_v1.h5
#CUDA_DEVICES_VISIBLE=0 python test/eval.py --file-in ../data/test_broken_train.txt --base-dir /home/lixiaoyu80/Datas/phone_attribute --out-file ../data/broken_train_test_result.txt  --record-file ../data/broken_train_test_result.csv  --modelpath /home/lixiaoyu80/models/mobilephone/broken_allian_best_ecaresnet101dv1.pth --cmd-type evalue 
#python test/eval.py --file_in ..\data\p2025_crops_broken.txt --base_dir D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops  --out_file ../data/broken_crops_p2025_mlp_result.txt  --record_file ../data/broken_crops_p2025_mlp_result.csv  --modelpath d:\models\phone_cls\broken_parts_best_mlp.pth --cmd_type evalue  

#python utils/plot_vis.py --file-in ../data/broken_p24_ecaresnet101dv2_result.csv --basename broken_ecaresnet101_pd --cmd-type plot
#python utils/plot.py --file-in ../data/res50_p20_result.txt --base-name p20_res50_prorg --cmd-type plot3data
#python utils/plot_vis.py --file-in ../data/broken_p24_ecaresnet101dv2_result.csv --out-file ../data/broken_p24_ecaresnet101v2_fpr.txt --cmd-type pdata
#python utils/plot_vis.py --file-in ../data/res50_online_result.csv --out-file ../data/res50_fpr_online.txt --cmd-type pdata
#python test/get_pr.py --file-in ../data/broken_p2025_ecaresnet101d_fuguan_result.csv --out-file ../data/broken_p2025_ecaresnet101d_fuguan_score2pr.csv --cmd-type evalue
#python test/get_pr.py --file-in ../data/broken_p2025_ecaresnet101d_fuguan_score2pr.csv  --name broken_p2025_ecaresnet101d_fuguan_score2pr --cmd-type plotscore
#python test/get_pr.py --file-in ../data/broken_p2025_ecaresnet101d_fuguan_score2pr.csv  --name broken_p2025_ecaresnet101d_fuguan_roc --cmd-type plotroc
#python test/get_pr.py --file-in ../data/broken_p2025_ecaresnet101d_fuguan_score2pr.csv  --name broken_p2025_ecaresnet101d_fuguan_score2pr-single --cmd-type single

#python test/get_pr.py --file-in ../data/mv1_p20_result.csv --out-file ../data/mv1_score2pr.csv --cmd-type evalue
#python test/get_pr.py --file-in ../data/mv1_score2pr.csv --name mv1_score2pr_roc --cmd-type plot
#***
CUDA_DEVICES_VISIBLE=0 python test/demo.py --file_in  ../data/val.txt  --cuda_use true --modelpath ../models/smogv5_best_resnet18.pth --img-dir /home/lg/Project/yolo-blacksmog/smog_train_v5/images --file_out result_val.txt
#CUDA_DEVICES_VISIBLE=0 python test/demo.py --file_in  ../data/phoneattribute_test.txt  --cuda_use true --modelpath /home/lixiaoyu80/models/broken_phone_model_v2_many_labels_v2.pb --img-dir /home/lixiaoyu80/Datas --file_out phonetest_result.txt


#****************
#CUDA_DEVICES_VISIBLE=0 python test/eval.py --base-dir  /home/lixiaoyu80/Develop/old_code/mnt/wangyakun5/broken_pro_new_data/data_upload_second_clean/rename_data_wyk1205 --file-in ../data/wyk_data.txt --out-file ../data/broken_org.txt  --record-file ../data/broken_org.csv  --modelpath /home/lixiaoyu80/models/broken_phone_model_v2_many_labels_v2.pb --cmd-type evalue
