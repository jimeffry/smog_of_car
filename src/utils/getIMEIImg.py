import cv2
import numpy as np
import requests
import json
import base64

def check_imei_luhn(numb_list):
        number = 0
        for i, num in enumerate(numb_list[:14]):
            num = int(num)
            if i % 2 == 0:
                number += num
            else:
                numTmp = (num * 2)
                if numTmp < 10:
                    number += numTmp
                else:
                    number += ((numTmp - 10) + 1)
        if number % 10 == 0:
            return '0'
        else:
            return str(10 - (number % 10))
            
class PhoneClassifybyWeb():
    def __init__(self):
        pass

    def __call__(self, image, sys_flag = 0):
        imageOrg = image.copy()
        image = cv2.imencode('.jpg', image)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        data = base64.b64encode(image).decode("utf-8")
        body = {"example": data}
        if sys_flag == 1:
            url = "http://jdaz-phonesystem.jd.local:2000/v1/PhoneClassify/1"
        elif sys_flag == 2:
            url = "http://jdaz-phonesystem.jd.local:2000/v1/PhoneIMEI/1"
        elif sys_flag == 3:
            url = "http://jdaz-phonesystem.jd.local:2000/v1/GeneralOCR/1"
        else:
            url = "http://jdaz-phonesystem.jd.local:2000/v1/PhoneSystem/1"
        try:
            response = requests.post(url, data = json.dumps(body))
            res = json.loads(response.text)
        except:
            return imageOrg, "网络传输失败，请检测网络状态！"

        output_strlist = ""
        returnArray = res["result"]
        if sys_flag == 3:
            imgOut, output_strlist = self.GeneralOCRPos(imageOrg, res["result"])
        else:
            imgOut = self.ImeiShow(imageOrg, res["result"])
        #********************加工输出 array
        for idx in range(0, len(returnArray)):
            item = returnArray[idx]
            if sys_flag == 2 or sys_flag == 0:
                if item['imeiNum'] == "-1" or item['imeiNum'] == 0 :
                    tmpstr = "图片中没有IMEI码"
                    output_strlist += tmpstr
                else:
                    for idx,imeiInfo in enumerate(item["imeiValue"]):
                        if len(imeiInfo['imei']) == 15 and imeiInfo['imei'][14] == check_imei_luhn(list(imeiInfo['imei'])) and imeiInfo['imei'][:2] in ["86", "35"]:
                            tmpstr = "<p> IMEI_%d 内容: %s , 概率值(取值范围为0.0到1.0): %.3f </p>" %(idx+1,imeiInfo['imei'],float(imeiInfo['prob']))
                            output_strlist += tmpstr
            elif sys_flag == 1 or sys_flag == 0:
                pass
                #phoneSceenClsProbs = float(item["brokenProb"])
                #tmpstr = "手机碎屏概率值(取值范围为0.0到1.0): %.3f" %(phoneSceenClsProbs)


                #output_strlist += tmpstr
            
        return imgOut, output_strlist
    
    def ImeiShow(self, img, returnArray):
        bboxList = []
        for imeiInfo in returnArray[0]["imeiValue"]:
            bbox = imeiInfo["bbox"]
            bboxList.append(bbox)
        #for brokenArea in returnArray[0]["brokenAreaBboxes"]
        #    bbox = brokenArea['bbox']
        #    bboxList.append(bbox)
        resImg = self.plot_QUAD(img, bboxList, c = (0, 0, 255))
        return resImg

    def GeneralOCRPos(self, imgOrg, ocr_list):
        ocr_res = ""
        bbox = []
        for ocr_info in ocr_list:
            line_text = ocr_info["text"]
            prob = round(np.mean(ocr_info["prob"]), 3)
            bbox.append(ocr_info["bbox"])
            ocr_res += line_text + "<br>"
        imgShow = self.plot_QUAD(imgOrg, bbox)
        return imgShow, ocr_res 

    def plot_QUAD(self, img, quads, line_width=2, c=(0, 0, 255)):
        tmp = np.array(img)
        for box in quads:
            x1, y1, x2, y2, x3, y3, x4, y4 = box

            cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, line_width)
            cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, line_width)
            cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, line_width)
            cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, line_width)
        return tmp
    
    def general_ocr_inference(imgOrg, ocr_list):
        ocr_res = ""
        bbox = []
        for ocr_info in ocr_list:
            line_text = ocr_info[0]
            prob = round(np.mean(ocr_info[1]), 3)
            bbox.append(ocr_info[2])
            ocr_res += line_text + "<br>"
        imgShow = self.plot_QUAD(imgOrg, bbox)
        return imgShow, ocr_res 

if __name__ == '__main__':
    import os
    savedir = "D:\Datas\mobilephone\phone_pre_cls\quality_imgs\p20bg"
    phoneClsHandler = PhoneClassifybyWeb()
    # imgFolder = r"D:\\数据集\\手机数据20210520\\PS2-0520\\"
    imgFolder = "D:\Datas\mobilephone\phone_detect\phone_crops\pall_crops\p20_crops_clear\\unbroken_imgs"
    for base_path, folder_list, file_list in os.walk(imgFolder):     
        for file in file_list:
            # print(file)
            imgFile = os.path.join(base_path, file)
            img = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), cv2.IMREAD_COLOR)
            imgOut, imei_str = phoneClsHandler(img, sys_flag = 2)
            if "图片中没有IMEI码" == imei_str:
                continue
            else:
                print(file + "Have IMEI!", imei_str)
                savepath = os.path.join(savedir,file.strip())
                cv2.imwrite(savepath, img)