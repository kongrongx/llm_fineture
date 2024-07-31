


import json
import argparse


def convert_qwen2(in_file, out_file, content_in=None, content_out=None):
    # 打开文件并逐行读取
    new_fp = []
    with open(in_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除行尾的换行符和可能的空格
            line = line.strip()
            # 检查行是否为空或者是否是注释等，如果是则跳过
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            try:
                # 尝试将行内容解析为JSON对象
                data = json.loads(line)

                # 输出解析后的JSON对象
                # print(data)

                s = {"messages":data.get('conversations')}
                # print(s)
                new_fp.append(s)

                # if content_in in data and content_out in data:
                #     s = {'messages': [{'role': 'user', 'content': '%s' % data[content_in]},
                #                            {'role': 'assistant', 'content': '%s' % data[content_out]}]
                #          }
                #     new_fp.append(s)
                # else:
                #     continue


            except json.JSONDecodeError as e:
                print(f"解析错误：{e}")

        with open(out_file, 'wt', encoding='utf-8') as fout:
            for s in new_fp:
                print(s)
                sample = s
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')




if __name__ == '__main__':
    in_file = 'data/self_cognition/dev.json'
    out_file = 'data/self_cognition/new_dev.json'


    convert_qwen2(in_file, out_file)






