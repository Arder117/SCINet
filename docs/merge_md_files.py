import os

def merge_md_files(input_dir, output_file):
    # 获取所有 .md 文件
    md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    
    # 如果没有 .md 文件，则输出提示并返回
    if not md_files:
        print("No .md files found in the directory.")
        return
    
    # 排序文件（按字母顺序）
    md_files.sort()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for md_file in md_files:
            md_file_path = os.path.join(input_dir, md_file)
            with open(md_file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                if content:  # 仅在文件有内容时合并
                    outfile.write(f"# {md_file}\n\n")  # 添加文件名作为标题
                    outfile.write(content)  # 写入文件内容
                    outfile.write("\n\n---\n\n")  # 添加分隔符
                else:
                    print(f"Warning: {md_file} is empty and was skipped.")
    
    print(f"All .md files have been merged into {output_file}")

# 调用函数合并文件
input_directory = r'C:\Users\ZhaoHao\OneDrive\Desktop\SCINet\docs'  # .md 文件所在的文件夹
output_file = r'C:\Users\ZhaoHao\OneDrive\Desktop\SCINet\docs\merged.md'  # 合并后的输出文件
merge_md_files(input_directory, output_file)