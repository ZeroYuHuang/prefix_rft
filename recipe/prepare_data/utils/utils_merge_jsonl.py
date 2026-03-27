import glob

file_dir="/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/math/distill_DeepSeek-R1-Distill-Qwen-32B"

# 定义输入文件模式和输出文件名
input_pattern = f"{file_dir}/*.jsonl"
output_file = f"{file_dir}/train.jsonl"

# 获取所有匹配的文件
file_list = sorted(glob.glob(input_pattern))

# 使用集合去重
unique_lines = set()

# 遍历所有文件并读取内容
print(file_list)
for file in file_list:
    with open(file, "r", encoding="utf-8") as infile:
        for line in infile:
            unique_lines.add(line)
        print(len(unique_lines))

# 写入去重后的内容
with open(output_file, "w", encoding="utf-8") as outfile:
    for line in unique_lines:
        outfile.write(line)

print(f"合并完成（去重），结果保存在 {output_file}")