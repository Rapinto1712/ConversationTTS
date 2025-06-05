import random


def read_scp(file_path):
    """Read the .scp file and return a dictionary with id as key and line as value."""
    entries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id, path = parts
                entries[id] = line.strip()
    return entries


def split_entries(ids, train_ratio=0.99, valid_ratio=0.005):
    """Split ids into train, validation, and test sets based on given ratios."""
    random.shuffle(ids)
    total = len(ids)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_ids = ids[:train_end]
    valid_ids = ids[train_end:valid_end]
    test_ids = ids[valid_end:]

    return train_ids, valid_ids, test_ids


def write_split_entries(id_splits, original_dict, base_filename):
    """Write entries to separate files based on the splits."""
    filenames = {
        'train': f'/home/ydc/musicllm/v1/tools/my_utils/data/train/{base_filename}.scp',
        'valid': f'/home/ydc/musicllm/v1/tools/my_utils/data/val/{base_filename}.scp',
        'test': f'/home/ydc/musicllm/v1/tools/my_utils/data/test/{base_filename}.scp'
    }

    for split_name, ids in id_splits.items():
        with open(filenames[split_name], 'w', encoding='utf-8') as file:
            for id in sorted(ids):  # Sort to maintain order
                if id in original_dict:
                    file.write(original_dict[id] + '\n')


# File paths
mp3_scp_file = 'common_music_mp3.scp'
texts_scp_file = 'right_music_texts.scp'

# Read the files into dictionaries
mp3_entries = read_scp(mp3_scp_file)
texts_entries = read_scp(texts_scp_file)

# Find common ids and shuffle them
common_ids = list(set(mp3_entries.keys()) & set(texts_entries.keys()))
random.seed(42)  # Set seed for reproducibility
train_ids, valid_ids, test_ids = split_entries(common_ids)

# Prepare splits for writing
id_splits = {
    'train': train_ids,
    'valid': valid_ids,
    'test': test_ids
}

# Write the splits to new files
write_split_entries(id_splits, mp3_entries, 'wav')
write_split_entries(id_splits, texts_entries, 'text')

print("Entries have been randomly split and written to separate train, valid, and test files.")