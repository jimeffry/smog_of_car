import numpy as np
import json
import os
import tqdm
import cv2

def load_json(json_dir,fileout,imgdir):
    breakcnt=0
    fw=open(fileout,"w") 
    fcnts = os.listdir(json_dir)
    label = 1
    for i in tqdm.tqdm(range(len(fcnts))):
        tmp = fcnts[i].strip()
        json_file = os.path.join(json_dir,tmp)
        imagename = tmp[:-5]
        load_f = open(json_file,'r',encoding='UTF-8')
        t_json_obj = json.load(load_f)
        features =t_json_obj['markResult']['features']
        coordinate_list = []
        imgpath = os.path.join(imgdir,imagename)
        img = cv2.imread(imgpath)
        for obj in features:
            if  not 'title'  in  obj :
                # print("err:",obj)
                continue
            title= obj['title']
            #包含brokenscreen 为碎屏区域 包含 phone为手机区域
            if title.find('brokenscreen')!= -1:
                list_break = obj['geometry']['coordinates']
                breakcnt = breakcnt + 1
                # for tmp_list in list_break:
                list_break_arr = np.array(list_break[0],dtype=np.int32)
                points_array=list_break_arr.reshape((-1,2))
                max_list=points_array.max(axis=0)
                min_list=points_array.min(axis=0)
                polygon_mask = cv2.fillPoly(img,[list_break_arr], 255)
                cv2.rectangle(img, (min_list[0],min_list[1]),(max_list[0],max_list[1]), (0,0,255), 20)
                min_list = min_list.tolist()
                max_list = max_list.tolist()
                coordinate_list.extend(min_list)
                coordinate_list.extend(max_list)
                coordinate_list.append(label)
                # print(coordinate_list)
        if len(coordinate_list)>=1:
            img = cv2.resize(img,(640,640))
            cv2.imshow('src',img)
            cv2.waitKey(0)
            coordinate_list = list(map(str,coordinate_list))
            points_str = ",".join(coordinate_list)
            fw.write("{},{}\n".format(imagename,points_str))
    print(breakcnt)
    fw.close()

if __name__ == "__main__":
     annodir = 'D:\Datas\mobilephone\\rename_data_wyk1205\\anno_rename'
     fileout = '..\data\\broken_boxes_t.txt'
     load_json(annodir,fileout,'D:\Datas\mobilephone\\rename_data_wyk1205\wyk')