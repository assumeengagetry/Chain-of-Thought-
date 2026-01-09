import openai
import time
import os

# 1. 设置 Kimi API
# 从环境变量读取 Kimi API Key
client = openai.OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),  
    base_url="https://api.moonshot.cn/v1"   
)

# 2. 定义测试问题列表
test_questions = [
    "一个篮子里有15个苹果。如果小明拿走了3个，然后小红又放入了比现在篮子里苹果数多一半的苹果，最后篮子里有多少个苹果？",
    "如果三只猫三天能捉三只老鼠，那么九只猫九天能捉多少只老鼠？",
    "我前面有两个人，后面有两个人，我们这一排一共有多少人？",
]

# 3. 定义提问模板
zero_shot_template = "{question} 请直接给出最终答案。"
cot_template = "{question} 让我们一步一步地思考。"

def ask_kimi(prompt, model="moonshot-v1-8k"):
    """调用 Kimi 模型并返回其回答"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度保证输出更确定，减少随机性
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用出错: {e}"

# 4. 主实验循环
def run_experiment():
    results = []
    for i, question in enumerate(test_questions):
        print(f"\n--- 正在处理问题 {i+1} ---")
        print(f"问题: {question}")

        # 零样本提问
        zero_shot_prompt = zero_shot_template.format(question=question)
        zero_shot_answer = ask_kimi(zero_shot_prompt)
        print(f"零样本回答:\n{zero_shot_answer}")

        # 等待一下，避免API速率限制
        time.sleep(1)

        # 思维链提问
        cot_prompt = cot_template.format(question=question)
        cot_answer = ask_kimi(cot_prompt)
        print(f"CoT回答:\n{cot_answer}")

        # 记录结果
        results.append({
            "question": question,
            "zero_shot_prompt": zero_shot_prompt,
            "zero_shot_answer": zero_shot_answer,
            "cot_prompt": cot_prompt,
            "cot_answer": cot_answer,
        })

        # 等待一下
        time.sleep(1)

    return results

# 5. 执行实验并打印结果
if __name__ == "__main__":
    all_results = run_experiment()
    # 保存结果为JSON文件
    import json
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

#sk-rykmzswJ5XaNmgx66mhI7cIyDXHqEvOxmh3NpgCCKOeH12X1