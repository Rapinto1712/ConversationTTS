import os


def read_and_flatten_text(file_path):
    """Reads a text file and returns its content as a single line."""
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines and join them into a single string separated by spaces
        return ' '.join(line.strip() for line in file)


def process_scp_file(input_scp, output_file):
    """Processes an scp file to read texts and write them in the desired format."""
    with open(input_scp, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            id, path = line.strip().split(maxsplit=1)
            text = read_and_flatten_text(path)
            # Write in "id text" format
            outfile.write(f"{id} {text}\n")


# Define paths
input_scp_path = 'common_music_texts.scp'
output_file_path = 'right_music_texts.scp'

# Process the files
process_scp_file(input_scp_path, output_file_path)

print("Processing complete.")