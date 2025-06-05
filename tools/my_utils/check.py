import os 
import glob 
import json
def get_language_id(): 
    names = glob.glob("/data4/MusicDataSliced_QQMusicWeb_2/*")
    ans = []
    for name in names:
        bs_name = os.path.basename(name)
        fls = glob.glob(f"{name}/*.json")
        for itm in fls:
            print('itm ', itm)
            try: 
                with open(itm, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    language = data['Language']
                    print('bs_name ', bs_name, language)
                    if language == '英语':
                        ans.append(bs_name)
            except:
                pass
    return ans 


for subdir, dirs, files in os.walk('/data4/MusicDataSliced_QQMusicWeb_2'):
    print(subdir)
    assert 1==2
# print(ans,len(ans))

