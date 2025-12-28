from openai.types.chat import ChatCompletionUserMessageParam

from LLM_Config import client, MODEL_NAME

# 基于prompt生成文本
def get_completion(prompt, model=MODEL_NAME):                               # 使用LLM_Config中的模型
    messages = [
        ChatCompletionUserMessageParam(role="user", content=prompt)         # 将prompt作为用户输入
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def simple_prompt_chat():
    user_prompt = 'Hello!'
    print(f"Prompt: {user_prompt}")
    response = get_completion('Hello!')
    print(f"Response: {response}")

def chat_with_prompt(prompt):
    print(f"Prompt: {prompt}")
    response = get_completion(prompt)
    print(f"Response: {response}")

# 第一版提示词，仅包含任务描述和用户输入
def nlu_prompt_v1():
    # 任务描述
    instruction = """
    你的任务是识别用户对手机流量套餐产品的选择条件。
    每种流量套餐产品包含三个属性：名称，月费价格，月流量。
    根据用户输入，识别用户在上述三种属性上的倾向。
    """

    # 用户输入
    input_text = """办一个100G的套餐"""

    # prompt模板。instruction和input_text会被替换为上面的内容
    prompt = f"""
    {instruction}
    
    用户输入：
    {input_text}
    """

    print(f"完整提示词：{prompt}")
    return prompt

# 第二版提示词，包含任务描述，用户输入和输出格式
def nlu_prompt_v2():
    # 任务描述
    instruction = """
    你的任务是识别用户对手机流量套餐产品的选择条件。
    每种流量套餐产品包含三个属性：名称，月费价格，月流量。
    根据用户输入，识别用户在上述三种属性上的倾向。
    """

    # 用户输入
    input_text = """办一个100G的套餐"""

    # 输出格式
    output_format = """以JSON格式输出"""

    # 稍微调整下咒语，加入输出格式
    prompt = f"""
    {instruction}
    
    {output_format}

    用户输入：
    {input_text}
    """

    print(f"完整提示词：{prompt}")
    return prompt


# 第三版提示词，包含任务描述，用户输入和更精细的输出格式
def nlu_prompt_v3():
    # 任务描述
    instruction = """
    你的任务是识别用户对手机流量套餐产品的选择条件。
    每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
    根据用户输入，识别用户在上述三种属性上的倾向。
    """

    # 用户输入
    # input_text = "办个100G以上的套餐"
    # input_text = "我要无限量套餐"
    input_text = "有没有便宜的套餐"

    # 输出格式增加了各种定义，约束
    output_format = """
    以JSON格式输出。
    1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；
    
    2. price字段的取值为一个结构体 或 null，包含两个字段：
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型
    
    3. data字段的取值为一个结构体 或 null，包含两个字段
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型或string类型，string类型只能是'无上限'
    
    4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
    （1）结构体中以"ordering = descend"表示按降序排序，以"value"字段存储待排序的字段
    （2）结构体中以"ordering = ascend"表示按升序排序，以"value"字段存储待排序的字段
    
    输出中只包含用户提及的字段，不要猜测任何用户未提及的字段，不输出值为null的字段    
    """

    prompt = f"""
    {instruction}

    {output_format}

    用户输入：
    {input_text}
    """

    print(f"完整提示词：{prompt}")
    return prompt


# 第四版提示词，包含任务描述，用户输入，更精细的输出格式和部分例子
def nlu_prompt_v4():
    # 任务描述
    instruction = """
    你的任务是识别用户对手机流量套餐产品的选择条件。
    每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
    根据用户输入，识别用户在上述三种属性上的倾向。
    """

    # input_text = "有没有便宜的套餐"
    # input_text = "有没有土豪套餐"
    # input_text = "办个200G的套餐"
    # input_text = "有没有流量大的套餐"
    # input_text = "200元以下，流量大的套餐有啥"
    input_text = "你说那个10G的套餐，叫啥名字"

    examples = """
    便宜的套餐：{"sort":{"ordering"="ascend","value"="price"}}
    有没有不限流量的：{"data":{"operator":"==","value":"无上限"}}
    流量大的：{"sort":{"ordering"="descend","value"="data"}}
    100G以上流量的套餐最便宜的是哪个：{"sort":{"ordering"="ascend","value"="price"},"data":{"operator":">=","value":100}}
    月费不超过200的：{"price":{"operator":"<=","value":200}}
    就要月费180那个套餐：{"price":{"operator":"==","value":180}}
    经济套餐：{"name":"经济套餐"}
    """

    # 输出格式增加了各种定义，约束
    output_format = """
    以JSON格式输出。
    1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

    2. price字段的取值为一个结构体 或 null，包含两个字段：
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型

    3. data字段的取值为一个结构体 或 null，包含两个字段
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型或string类型，string类型只能是'无上限'

    4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
    （1）结构体中以"ordering = descend"表示按降序排序，以"value"字段存储待排序的字段
    （2）结构体中以"ordering = ascend"表示按升序排序，以"value"字段存储待排序的字段

    输出中只包含用户提及的字段，不要猜测任何用户未提及的字段，不输出值为null的字段    
    """

    # 加入例子
    prompt = f"""
    {instruction}

    {output_format}
    
    例如：
    {examples}

    用户输入：
    {input_text}
    """

    print(f"完整提示词：{prompt}")
    return prompt


# 第五版提示词，包含任务描述，用户输入，更精细的输出格式，部分例子。并在prompt中加入上下文
def nlu_prompt_v5():
    # 任务描述
    instruction = """
    你的任务是识别用户对手机流量套餐产品的选择条件。
    每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
    根据对话上下文，识别用户在上述三种属性上的倾向。识别结果要包含整个对话的信息。
    """

    # 输出描述
    output_format = """
    以JSON格式输出。
    1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

    2. price字段的取值为一个结构体 或 null，包含两个字段：
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型

    3. data字段的取值为一个结构体 或 null，包含两个字段
    （1） operator, string类型，取值范围：'<=' （小于等于）, '>=' （大于等于）， '=='（等于）
    （2） value, int 类型或string类型，string类型只能是'无上限'

    4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
    （1）结构体中以"ordering = descend"表示按降序排序，以"value"字段存储待排序的字段
    （2）结构体中以"ordering = ascend"表示按升序排序，以"value"字段存储待排序的字段

    输出中只包含用户提及的字段，不要猜测任何用户未提及的字段，不输出值为null的字段    
    """

    # 多轮对话的例子
    examples = """
    客服：有什么可以帮您
    用户：100G套餐有什么
    
    {"data":{"operator":">=","value":100}}
    
    客服：有什么可以帮您
    用户：100G套餐有什么
    客服：我们现在有无限套餐，不限流量，月费300元
    用户：太贵了，有200元以内的不
    
    {"data":{"operator":">=","value":100},"price":{"operator":"<=","value":200}}
    
    客服：有什么可以帮您
    用户：便宜的套餐有什么
    客服：我们现在有经济套餐，每月50元，10G流量
    用户：100G以上的有什么
    
    {"data":{"operator":">=","value":100},"sort":{"ordering"="ascend","value"="price"}}
    
    客服：有什么可以帮您
    用户：100G以上的套餐有什么
    客服：我们现在有畅游套餐，流量100G，月费180元
    用户：流量最多的呢
    
    {"sort":{"ordering"="descend","value"="data"},"data":{"operator":">=","value":100}}
    """

    input_text = "哪个便宜"
    # input_text = "无限量那个多少钱"
    # input_text = "流量最大的多少钱"

    # 多轮对话上下文
    # context_list = []
    # context_list.append(prompt)
    # '\n'.join(context_list)
    context = f"""
    客服：有什么可以帮您
    用户：有什么100G以上的套餐推荐
    客服：我们有畅游套餐和无限套餐，您有什么价格倾向吗
    用户：{input_text}
    """

    # 加入例子
    prompt = f"""
    {instruction}

    {output_format}

    {examples}

    {context}
    """

    print(f"完整提示词：{prompt}")
    return prompt


def main():
    # print("1. 首先测试使用最简单的提示词来聊天")
    # simple_prompt_chat()
    #
    # print("2. 测试使用第一版NLU提示词，仅包含任务描述和用户输入")
    # chat_with_prompt(prompt=nlu_prompt_v1())
    #
    # print("3. 测试使用第二版NLU提示词，包含任务描述，用户输入和输出格式")
    # chat_with_prompt(prompt=nlu_prompt_v2())
    #
    # print("4. 测试使用第三版NLU提示词，包含任务描述，用户输入和更精细定义的输出格式")
    # chat_with_prompt(prompt=nlu_prompt_v3())
    #
    # print("5. 测试使用第四版NLU提示词，包含任务描述，用户输入，更精细定义的输出格式和部分例子")
    # chat_with_prompt(prompt=nlu_prompt_v4())

    print("6. 测试使用第五版NLU提示词，包含任务描述，用户输入，更精细定义的输出格式，部分例子。并包含对话上下文")
    chat_with_prompt(prompt=nlu_prompt_v5())

if __name__ == '__main__':
    main()