import json
import copy
from dotenv import load_dotenv, find_dotenv
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionAssistantMessageParam

from LLM_Config import client, MODEL_NAME
from typing import List, Union
_ = load_dotenv(find_dotenv())

# ===== 系统级提示：告诉模型它的任务与输出格式 =====
instruction = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
根据用户输入，识别用户在上述三种属性上的倾向。
"""

# ===== 输出格式说明：让模型以结构化 JSON 返回 =====
output_format = """
以JSON格式输出。
1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

2. price字段的取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型

3. data字段的取值为取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型或string类型，string类型只能是'无上限'

4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
(1) 结构体中以"ordering"="descend"表示按降序排序，以"value"字段存储待排序的字段
(2) 结构体中以"ordering"="ascend"表示按升序排序，以"value"字段存储待排序的字段

只输出中只包含用户提及的字段，不要猜测任何用户未直接提及的字段。
DO NOT OUTPUT NULL-VALUED FIELD! 不要使用```json```包装返回的json值，确保输出能被json.loads加载。
"""

# ===== 示例：帮助模型理解如何把自然语言映射到 JSON 结构 =====
examples = """
便宜的套餐：{"sort":{"ordering"="ascend","value"="price"}}
有没有不限流量的：{"data":{"operator":"==","value":"无上限"}}
流量大的：{"sort":{"ordering"="descend","value"="data"}}
100G以上流量的套餐最便宜的是哪个：{"sort":{"ordering"="ascend","value"="price"},"data":{"operator":">=","value":100}}
月费不超过200的：{"price":{"operator":"<=","value":200}}
就要月费180那个套餐：{"price":{"operator":"==","value":180}}
经济套餐：{"name":"经济套餐"}
"""

class NLU:
    """
    自然语言理解（NLU）模块
    负责把用户输入的自然语言转换成结构化的语义表示（JSON）
    """
    def __init__(self):
        # 构造最终发给大模型的prompt: 指令 + 输出格式 + 示例 + 占位符 __INPUT__
        self.prompt_template = f"{instruction}\n\n{output_format}\n\n{examples}\n\n用户输入：\n__INPUT__"

    def _get_completion(self, prompt, model=MODEL_NAME):
        """
        调用大模型 API 并返回解析后的 JSON 结果

        参数:
            prompt (str): 发送给模型的完整提示
            model (str): 模型名称，默认 deepseek-v3-250324

        返回:
            dict: 过滤掉空值后的语义结构
        """
        messages = [
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # 确定性最高，避免随机性
        )
        # 解析模型返回的JSON字符串并移除空值字段
        semantics = json.loads(response.choices[0].message.content)
        return {k: v for k, v in semantics.items()}

    def parse(self, user_input):
        """
        对外暴露的解析接口

        参数:
            user_input (str): 用户原始输入

        返回:
            dict: 结构化的语义结果
        """
        prompt = self.prompt_template.replace("__INPUT__", user_input)
        return self._get_completion(prompt)

class DST:
    """
    对话状态跟踪器 (Dialog State Tracker)
    在多轮对话中持续更新并维护用户的目标状态
    """
    def __init__(self):
        # 初始化时无需额外操作
        pass

    def update(self, state, nlu_semantics):
        """
        根据 NLU 结果更新对话状态

        参数:
            state (dict): 当前对话状态
            nlu_semantics (dict): NLU 解析出的最新语义

        返回:
            dict: 更新后的对话状态

        业务规则:
            1. 如果用户直接指定套餐名称，则清空之前所有条件
            2. 如果用户要求排序，且之前存在同字段的精确匹配(==)，则删除该精确匹配
            3. 其余情况直接合并新语义到状态
        """
        # 规则1：指定套餐名称时清空历史条件
        if "name" in nlu_semantics:
            state.clear()
        # 规则2：排序与精确匹配冲突时删除精确匹配
        if "sort" in nlu_semantics:
            slot = nlu_semantics["sort"]["value"]           # 排序字段
            if slot in state and state[slot]["operator"] == "==":
                del state[slot]

        # 合并新语义
        for k, v in nlu_semantics.items():
            state[k] = v
        return state

class MockedDB:
    """
    模拟数据库
    提供套餐数据的存储与检索功能
    """
    def __init__(self):
        # 预定义套餐数据
        self.data = [
            {"name": "经济套餐", "price": 50, "data": 10, "requirement": None},
            {"name": "畅游套餐", "price": 180, "data": 100, "requirement": None},
            {"name": "无限套餐", "price": 300, "data": 1000, "requirement": None},
            {"name": "校园套餐", "price": 150, "data": 200, "requirement": "在校生"},
        ]

    def retrieve(self, **kwargs):
        """
        根据用户状态与查询条件检索套餐

        参数:
         **kwargs: 查询条件，可包含 price、data、name、sort 等

        返回:
         list: 符合条件的套餐列表，已按指定字段排序
        """
        records = []

        for r in self.data:
            select = True
            # 检查套餐限制条件（如校园套餐仅限在校生）
            if r["requirement"]:
                if "status" not in kwargs or kwargs["status"] != r["requirement"]:
                    continue
            # 逐条匹配用户条件
            for k, v in kwargs.items():
                if k == "sort":                 # 排序字段不参与过滤
                    continue
                # 处理“无上限”流量特殊值
                if k == "data" and v["value"] == "无上限":
                    if r[k] != 1000:
                        select = False
                        break
                # 处理带比较运算符的条件
                elif "operator" in v:
                    if not eval(str(r[k]) + v["operator"] + str(v["value"])):
                        select = False
                        break
                # 处理精确匹配
                elif str(r[k]) != str(v):
                    select = False
                    break
            if select:
                records.append(r)

        # 结果为空或仅一条时直接返回
        if len(records) <= 1:
            return records

        # 默认按价格升序排序
        key = "price"
        reverse = False
        # 若用户指定排序字段与顺序则覆盖默认值
        if "sort" in kwargs:
            key = kwargs["sort"]["value"]
            reverse = kwargs["sort"]["ordering"] == "descend"
        return sorted(records, key=lambda x: x[key], reverse=reverse)

class DialogManager:
    """
    对话管理器
    串联NLU、DST、DB与LLM，实现完整的多轮对话流程
    """
    def __init__(self, prompt_templates):
        # 初始化对话状态
        self.state = {}
        # 初始化ChatGPT的session，设置系统人设
        self.session: List[Union[
            ChatCompletionUserMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionAssistantMessageParam,
        ]] = [
            ChatCompletionSystemMessageParam(role="system", content="你是一个手机流量套餐的客服代表，你叫小达。可以帮助用户选择最合适的流量套餐产品。"),
        ]
        # 初始化各子模块
        self.nlu = NLU()                                # 自然语言理解
        self.dst = DST()                                # 对话状态跟踪
        self.db = MockedDB()                            # 数据检索
        self.prompt_templates = prompt_templates        # 话术模板

    def _wrap(self, user_input, records):
        """
        根据检索结果拼装最终发给 ChatGPT 的 prompt

        参数:
            user_input (str): 用户原始输入
            records (list): 检索到的套餐列表

        返回:
            str: 拼装好的 prompt
        """
        if records:
            # 有匹配套餐时使用推荐模板
            prompt = self.prompt_templates["recommend"].replace("__INPUT__", user_input)
            r = records[0]
            for k, v in r.items():
                prompt = prompt.replace(f"__{k.upper()}__", str(v))
        else:
            prompt = self.prompt_templates["not_found"].replace("__INPUT__", user_input)
            for k, v in self.state.items():
                if "operator" in v:
                    prompt = prompt.replace(f"__{k.upper()}__", v["operator"]+str(v["value"]))
                else:
                    prompt = prompt.replace(f"__{k.upper()}__", str(v))
        return prompt

    def _call_chatgpt(self, prompt, model=MODEL_NAME):
        session = copy.deepcopy(self.session)
        session.append(ChatCompletionUserMessageParam(role="user", content=prompt))
        response = client.chat.completions.create(
            model=model,
            messages=session,
            temperature=0
        )
        return response.choices[0].message.content

    def run(self, user_input):
        # 调用NLU获得语义解析
        semantics = self.nlu.parse(user_input)
        print("===semantics===")
        print(semantics)

        # 调用DST更新多轮状态
        self.state = self.dst.update(self.state, semantics)
        print("===state==")
        print(self.state)

        # 根据状态检索DB，获得满足条件的候选
        records = self.db.retrieve(**self.state)

        # 拼装prompt调用chatgpt
        prompt_for_chatgpt = self._wrap(user_input, records)
        print("===prompt_for_chatgpt==")
        print(prompt_for_chatgpt)

        # 调用LLM获得回复
        response = self._call_chatgpt(prompt_for_chatgpt)

        # 将当前用户输入和系统回复维护入chatgpt的session
        self.session.append(ChatCompletionUserMessageParam(role="user", content=user_input))
        self.session.append(ChatCompletionAssistantMessageParam(role="assistant", content=response))
        return response

def main():
    # 1.0 加入指定情况下的回答模版，这样话术更专业
    prompt_templates = {
        "recommend": "用户说：__INPUT__ \n\n向用户介绍如下产品：__NAME__，月费__PRICE__元，每月流量__DATA__G。",
        "not_found": "用户说：__INPUT__ \n\n没有找到满足__PRICE__元价位__DATA__G流量的产品，询问用户是否有其他选择倾向。"
    }
    # user_input = "流量大的"

    # 2.0 增加约束，改变语气、口吻等风格
    # 定义语气要求。"NO COMMENTS. NO ACKNOWLEDGEMENTS."是常用 prompt，表示「有事儿说事儿，别 bb」
    # ext = "很口语，亲切一些。不用说“抱歉”。直接给出回答，不用在前面加“小达说：”。NO COMMENTS. NO ACKNOWLEDGEMENTS."
    # prompt_templates = {k: v+ext for k, v in prompt_templates.items()}
    # user_input = "300太贵了，200元以内有吗"

    # 3.0 实现统一口径
    ext = "\n\n遇到类似问题，请参照以下回答：\n问：流量包太贵了\n答：亲，我们都是全省统一价哦。"
    prompt_templates = {k: v+ext for k, v in prompt_templates.items()}
    user_input = "这流量包太贵了"

    dm = DialogManager(prompt_templates)

    response = dm.run(user_input)
    print("===response===")
    print(response)

if __name__ == "__main__":
    main()