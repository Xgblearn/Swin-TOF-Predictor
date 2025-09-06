import os
import glob


def extract_first_column(input_dir):
    """
    Extract the first column from all .txt files in a directory
    and save them to a sibling folder.

    Args:
        input_dir: directory containing .txt files
    """
    # create output directory (sibling of input)
    output_dir = os.path.join(os.path.dirname(input_dir), "target_signal_new")
    os.makedirs(output_dir, exist_ok=True)

    # list txt files
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))

    if not txt_files:
        print(f"Warning: no txt files found in '{input_dir}'")
        return

    processed_count = 0

    for file_path in txt_files:
        try:
            # read and extract first column
            extracted_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # strip newline/whitespace
                    cleaned_line = line.strip()
                    if cleaned_line:  # 确保行不为空
                        # split by comma
                        parts = cleaned_line.split(',')
                        # strip empty parts
                        parts = [p.strip() for p in parts if p.strip()]
                        if parts:
                            extracted_data.append(parts[0])

            # output file path (same filename)
            output_path = os.path.join(output_dir, os.path.basename(file_path))

            # write extracted data
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in extracted_data:
                    f_out.write(item + '\n')

            processed_count += 1
            print(f"Processed: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"\nDone. Processed {processed_count}/{len(txt_files)} files")
    print(f"Output dir: {output_dir}")


# 使用示例 - 替换为您的实际目录路径
if __name__ == "__main__":
    # Replace with your path
    input_directory = r"D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_test\target_signal"

    # 调用处理函数
    extract_first_column(input_directory)