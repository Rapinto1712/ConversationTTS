import os

def find_wav_files(root_dir):
    wav_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                abs_path = os.path.join(subdir, file)
                rel_name = os.path.splitext(file)[0]  # 获取文件名（不带扩展名）
                rel_name = rel_name.split('_')[-1]
                rel_name = str(subdir).split('/')[-1] + '_' + rel_name
                wav_files.append((rel_name, abs_path))
    wav_files.sort(key=lambda x: x[0])  # 根据相对名称排序
    return wav_files

def write_scp_file(scp_filename, wav_files):
    with open(scp_filename, 'w', encoding='utf-8') as f:
        for name, path in wav_files:
            f.write(f'{name} {path}\n')

if __name__ == '__main__':
    root_directory = '/data4/MusicDataSliced_QQMusicWeb_2'
    output_scp_file = 'music_texts.scp'

    print("开始收集WAV文件...")
    all_wav_files = find_wav_files(root_directory)
    print(f"找到 {len(all_wav_files)} 个WAV文件.")

    print("写入SCP文件...")
    write_scp_file(output_scp_file, all_wav_files)
    print(f"SCP文件已成功创建为 {output_scp_file}")