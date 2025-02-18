import csv

# formatted response files
files = [
    "llava_v15_13b_Real_formatting_results.csv"
    ]

for file in files:
    
    data = {}
    with open(file, 'r', encoding = "utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        cnt = 0
        for row in reader:
            if row['type'] not in data:
                data[row['type']] = []
            data[row['type']].append(row)

    print("----------------------")
    print(file)
    for type_category, rows in data.items():
        print(f"Type: {type_category}")
        acc_diff = 0
        correct_count = 0
        total_count = len(rows)
        for row in rows:
            if row['response'] == row['answer']:
                correct_count += 1
        ori_correct_ratio = correct_count / total_count * 100
        diff_ratio = 1 - ori_correct_ratio


        print(f"ori acc: {ori_correct_ratio:.1f}")
        acc_diff = ori_correct_ratio
        
        correct_count = 0
        total_count = len(rows)
        for row in rows:
            if row['new_response'] == row['new answer']:
                correct_count += 1
        cf_correct_ratio = correct_count / total_count * 100
        diff_ratio = 1 - cf_correct_ratio
        
        print(f"cf acc: {cf_correct_ratio:.1f}")
        acc_diff -= cf_correct_ratio
        print(f"diff(ori_acc - cf_acc): {acc_diff:.1f}")
        print()