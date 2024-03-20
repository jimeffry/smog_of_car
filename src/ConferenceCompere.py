#!/usr/bin/env python
# -*- coding: utf-8 -*-#
# @File          : ConferenceCompere.py
# @Created on    : 2021/02/18
# @Author        : mazheng23
# @Describe      : 适用于python3.8
'''
会议主持抽选
'''



import random
finish_name = ['申磊', '董宇青', '黄新苗','李晓宇']
all_names = ['申磊', '陈阳明', '赵英楠', '王洪国', '王菁梅','蔺小宁','任敬辉','武淼仑','朱林霞', '卜康亮', '曹羽', '李钰', '李净桦', '杨志军', '刘艺飞', '郑志德', '张燕' , '王维来', '马征','李晓宇', '赵统法', '黄新苗','董宇青', '秦树林','宋敏','','陈彬']
#print(len(all_names))
#下次周会种子
# random.seed(20210318)
luck_list = list(set(all_names) - set(finish_name))
print('总:',len(all_names),'已完成:',len(finish_name),"余:",len(luck_list))
luck_name= random.sample(luck_list, 2)

print("中奖人员：",luck_name)
