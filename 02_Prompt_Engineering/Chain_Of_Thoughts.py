from openai.types.chat import ChatCompletionUserMessageParam

from LLM_Config import client, MODEL_NAME

def get_completion(prompt, model=MODEL_NAME):
    messages = [
        ChatCompletionUserMessageParam(role="user", content=prompt)
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def build_prompt_for_think_step_by_step():
    instruction = """
    给定一段用户与手机流量套餐客服的对话，
    你的任务是判断客服介绍产品信息的准确性：
    
    当向用户介绍流量套餐产品时，客服人员必须准确提及产品名称、月费价格和月流量总量 上述信息缺失一项或多项，或信息与实时不符，都算信息不准确
    
    已知产品包括：
    
    经济套餐：月费50元，月流量10G
    畅游套餐：月费180元，月流量100G
    无限套餐：月费300元，月流量1000G
    校园套餐：月费150元，月流量200G，限在校学生办理
    """

    # 输出描述
    output_format = """
    以JSON格式输出。
    如果信息准确，输出：{"accurate":true}
    如果信息不准确，输出：{"accurate":false}
    """

    # context1 = """
    # 用户：你们有什么流量大的套餐
    # 客服：您好，我们现在正在推广无限套餐，每月300元就可以享受1000G流量，您感兴趣吗
    # """

    # context2 = """
    # 用户：有什么便宜的流量套餐
    # 客服：您好，我们有个经济型套餐，50元每月
    # """

    context3 = """
    用户：流量大的套餐有什么
    客服：我们推荐畅游套餐，180元每月，100G流量，大多数人都够用的
    用户：学生有什么优惠吗
    客服：如果是在校生的话，可以办校园套餐，150元每月，含200G流量
    
    Let’s think step by step
    """

    prompt = f"""
    {instruction}
    
    {output_format}
    
    对话记录：
    {context3}
    """

    return prompt

def build_prompt_for_knowledge():
    knowledge = "高尔夫球比赛的目标是在尽可能少的击球数下完成球场的18个洞。每个洞有一个标准杆数，球员的得分是击球数减去标准杆数。因此，高尔夫球比赛的一部分是试图获得比其他人更低的得分，而不是更高的得分。"
    prompt = f"""
    高尔夫球的一部分是试图获得比其他人更高的得分。是或否？

    知识：{knowledge}

    解释和答案：

    """
    return prompt

def main():
    # prompt = build_prompt_for_think_step_by_step()
    prompt = build_prompt_for_knowledge()
    response = get_completion(prompt=prompt)
    print(response)

if __name__ == '__main__':
    main()