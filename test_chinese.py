#encoding=utf-8
import os
import time
import pygame
import pandas as pd
list1=[]
b = range(3755)
for i in b:
	a = str(i)
	if len(a) == 1:
		c = '0000'+a
		chinese_dir = '../data/train/'+c
		list1.append(chinese_dir)
		if not os.path.exists(list1[i]):
		    os.mkdir(list1[i])
	elif len(a) == 2:
		c = '000'+a
		chinese_dir = '../data/train/'+c
		list1.append(chinese_dir)
		if not os.path.exists(list1[i]):
		    os.mkdir(list1[i])
	elif len(a) == 3:
		c = '00'+a
		chinese_dir = '../data/train/' +c
		list1.append(chinese_dir)
		if not os.path.exists(list1[i]):
		    os.mkdir(list1[i])
	else:
		c = '0'+a
		chinese_dir = '../data/train/'+c
		list1.append(chinese_dir)
		if not os.path.exists(list1[i]):
		    os.mkdir(list1[i])
pygame.init()
# start,end = (0x4E00, 0x9FA5) # 汉字编码范围
# for codepoint in range(int(start), int(end)):
mydata_txt = pd.read_csv('p/chinese.txt', sep=',', encoding='utf8')

list2 = [] 	
for i in mydata_txt:
	list2.append(i)
print(list2)
for i in range(len(list2)):
    for j in range(40):
    	if j%2==0:
    		font = pygame.font.Font("ziti/SIMLI.TTF", 64)
    	elif j%3==0:
    		font = pygame.font.Font("ziti/msyh.ttc", 64)
    	else:
    		font = pygame.font.Font("ziti/simsun.ttc", 64) 		
    	rtext = font.render(list2[i], True, (0,0,0), (255, 255, 255))
    	pygame.image.save(rtext, os.path.join(list1[i], str(time.time())+ ".png"))


