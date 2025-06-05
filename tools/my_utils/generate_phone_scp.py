import os
from tools.tokenizer.Text2Phone.Text2PhoneTokenizer import Text2PhoneTokenizer

def read_scp_file(scp_filename):
    with open(scp_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(' ', 1) for line in lines]

def get_text_file(wav_path):
    text_path = os.path.splitext(wav_path)[0] + '.normalized.txt'
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.readline().strip()  # 假设每行只有一条记录
        return text
    else:
        print(f"警告: 文本文件不存在 {text_path}")
        return None

def write_phone_sequence_to_file(output_filename, phone_sequences):
    with open(output_filename, 'w', encoding='utf-8') as f:
        for file_name, phone_seq in phone_sequences:
            f.write(f'{file_name} {phone_seq}\n')

def main():
    scp_file = '/home/rjh/UniAudio/egs/TTS/data/train/wav.scp'  # wav.scp文件路径
    output_file = '/home/rjh/UniAudio/egs/TTS/data/train/phone.scp'  # 输出文件路径

    T2P_tokenizer = Text2PhoneTokenizer()

    entries = read_scp_file(scp_file)
    phone_sequences = []

    for file_name, wav_path in entries:
        text = get_text_file(wav_path)
        if text is not None:
            phone_seq = 'SIL ' + ' '.join(T2P_tokenizer.get_phone_sequence(text))
            phone_sequences.append((file_name, phone_seq))

    write_phone_sequence_to_file(output_file, phone_sequences)
    print(f"输出文件已成功创建为 {output_file}")

if __name__ == '__main__':
    main()