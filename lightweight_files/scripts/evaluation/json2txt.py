import json
import argparse

# Here input the relative path of the prediction file
# e.g., 'outputs/predictions/WCCNet_kaist.json'
in_path = None 

parser = argparse.ArgumentParser("WCCNet json2txt parser")
parser.add_argument("-res", "--result", type=str, default=None)
args = parser.parse_args()
if args.result is not None:
    in_path = args.result
else:
    assert in_path is not None, "please set the relative path of the prediction file"
    
out_path = in_path.split('.')[0]+'.txt'

dicts = None
with open(in_path,'r') as f:
    dicts = json.load(f)
dicts.sort(key=lambda x:(x['image_id'],-x['score']))
txt_list = []
with open(out_path,'w') as f:
    for i in dicts:
        _line = [i['image_id'],i['bbox'][0],i['bbox'][1],i['bbox'][2],i['bbox'][3],i['score']]
        _line = f'{_line[0]},{_line[1]:.4f},{_line[2]:.4f},{_line[3]:.4f},{_line[4]:.4f},{_line[5]:.8f}'
        _line +='\n'
        f.write(_line)