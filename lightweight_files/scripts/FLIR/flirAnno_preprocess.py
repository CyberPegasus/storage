import sys
import os
import json
import xml.etree.ElementTree as ET

txt_read = ['datasets/alignedFLIR/align_train.txt','datasets/alignedFLIR/align_val.txt']
txt_write = [i.replace('align_','xml_') for i in txt_read]
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
anno_dir = 'datasets/alignedFLIR/annos/'# os.path.join(dataset_dir,"alignedFLIR/","annos/")
for index in range(len(txt_read)):
    with open(txt_read[index],'r') as fr:
        _lines = fr.readlines()
        _lines = [i.strip() for i in _lines]
        _lines = [i.split('_')[1] for i in _lines]
        _lines = [i+'.xml' for i in _lines]
        for i in _lines:
            assert os.path.exists(anno_dir+i), f'Not exists: {anno_dir+i}'
        with open(txt_write[index],'w') as fw:
            _lines = [i+'\n' for i in _lines]
            fw.writelines(_lines)