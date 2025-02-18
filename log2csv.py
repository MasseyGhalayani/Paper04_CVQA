import sys
import csv


def parse_log_with_header(log_filename):
    """
    Author : Massey_gh
    Reads the log file and returns a tuple (header, blocks).
    The header is the first non-empty line if it starts with "img_path".
    Then it groups the remaining lines into blocks of five lines each,
    where each block is assumed to have:
      [0]: Q1 response (e.g. "D")
      [1]: A marker like "Q1 ==>"
      [2]: Q2 response (e.g. "B")
      [3]: A marker like "Q2 ==>"
      [4]: A comma-separated "base row" that contains the core CSV columns.
    """
    header = None
    blocks = []
    with open(log_filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if lines and lines[0].lower().startswith("img_path"):
        header = lines[0]
        lines = lines[1:]

    i = 0
    while i < len(lines):
        if i + 4 < len(lines) and lines[i + 1].startswith("Q1") and lines[i + 3].startswith("Q2"):
            block = lines[i:i + 5]
            blocks.append(block)
            i += 5
        else:
            i += 1
    return header, blocks


def convert_block_to_csv_row(block):
    """
    Given a block of five lines:
      block[0]: Q1 response
      block[1]: "Q1 ==>"
      block[2]: Q2 response
      block[3]: "Q2 ==>"
      block[4]: The base row (a comma-separated line, may have extra trailing commas)
    This function returns a new CSV row string where the Q1 and Q2 responses are appended
    as the final two columns (after removing trailing commas from the base row).
    """
    q1_response = block[0].strip()
    q2_response = block[2].strip()

    base_row = block[4].rstrip(',')
    final_row = f"{base_row},{q1_response},{q2_response}"
    return final_row


def process_log_to_csv(log_filename, output_csv):
    header, blocks = parse_log_with_header(log_filename)
    rows = [convert_block_to_csv_row(block) for block in blocks]

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header.split(','))
        else:
            writer.writerow(
                ["img_path", "query", "answer", "new query", "new answer", "type", "response", "new_response"])
        for row in rows:
            writer.writerow(row.split(','))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_log_to_csv.py <slurm_log_file.out> <output.csv>")
        sys.exit(1)
    log_file = sys.argv[1]
    output_csv = sys.argv[2]
    process_log_to_csv(log_file, output_csv)
    print(f"CSV output saved to {output_csv}")
