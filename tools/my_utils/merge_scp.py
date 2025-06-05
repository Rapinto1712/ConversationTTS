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

def find_common_ids(dict1, dict2):
    """Find common ids between two dictionaries."""
    return set(dict1.keys()) & set(dict2.keys())

def write_common_entries(common_ids, original_dict, output_file):
    """Write common entries to the output file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for id in sorted(common_ids):  # Sort to maintain order
            if id in original_dict:
                file.write(original_dict[id] + '\n')

# File paths
mp3_scp_file = 'music_mp3.scp'
texts_scp_file = 'music_texts.scp'

# Output files
output_mp3_scp = 'common_music_mp3.scp'
output_texts_scp = 'common_music_texts.scp'

# Read the files into dictionaries
mp3_entries = read_scp(mp3_scp_file)
texts_entries = read_scp(texts_scp_file)

# Find common ids
common_ids = find_common_ids(mp3_entries, texts_entries)

# Write the common entries to new files
write_common_entries(common_ids, mp3_entries, output_mp3_scp)
write_common_entries(common_ids, texts_entries, output_texts_scp)

print(f"Common entries have been written to {output_mp3_scp} and {output_texts_scp}.")