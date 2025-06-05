import os 
# 打开并读取文件
f_r = open('/data6/ydc/exp_data/data/train/texts.scp')
f_w = open('/data6/ydc/exp_data/data/train/text.scp', 'w')

def rd(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 将每一行合并为一行，并去除换行符
    merged_line = ' '.join(line.strip() for line in lines)
    return merged_line

for line in f_r:
    ans = line.strip().split(' ')
    name = ans[0]
    c = rd(ans[1])
    f_w.write(name+' '+c+'\n')


