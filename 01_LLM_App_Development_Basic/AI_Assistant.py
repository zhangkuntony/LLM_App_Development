# 构建AI工具箱类
from typing import List, Union
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)
from LLM_Config import client, MODEL_NAME

class AIAssistant:
    """多功能AI助手"""
    def __init__(self, assistant_client):
        self.client = assistant_client
        self.conversation_history = []          # 记录对话历史

    def chat(self, user_message, role="助手", save_history=True):
        """基础对话功能"""

        # 明确指定消息列表的类型
        messages: List[Union[
            ChatCompletionUserMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionAssistantMessageParam
        ]] = [
            ChatCompletionSystemMessageParam(role="system", content=f"你是一个友好的AI{role}，用简洁、有用的方式回答问题。")
        ]

        # 添加历史对话（最近5轮）
        if self.conversation_history:
            messages.extend(self.conversation_history[-10:])            # 保持上下文

        messages.append(ChatCompletionUserMessageParam(role="user", content=user_message))

        print('当前的对话历史', messages)

        # 调用AI
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        ai_reply = response.choices[0].message.content

        # 保存对话历史
        if save_history:
            self.conversation_history.extend([
                ChatCompletionUserMessageParam(role="user", content=user_message),
                ChatCompletionAssistantMessageParam(role="assistant", content=ai_reply)
            ])

        return ai_reply

    def translate(self, text, target_language="英文"):
        """翻译功能"""
        prompt = f"请根据以下文本翻译层{target_language}，只输出翻译结果：\n{text}"
        return self.chat(prompt, role="翻译专家", save_history=False)

    def code_helper(self, description):
        """代码助手"""
        prompt = f"请根据以下描述写出Python代码，并添加注释说明：\n{description}"
        return self.chat(prompt, role="编程导师", save_history=False)

    def creative_writing(self, topic, style="幽默"):
        """创意写作"""
        prompt = f"请以'{style}'的风格，围绕'{topic}'这个主题写一段有趣的内容（100字左右）。"
        return self.chat(prompt, role="创意作家", save_history=False)

    def clean_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话离职已清空")

def run_basic_chat(ai_assistant):
    # 1. 基础对话
    print("基础对话测试：")
    response = ai_assistant.chat('你好，我想学习AI开发，有什么建议吗？')
    print(f"AI回复：{response}\n")

def run_translate(ai_assistant):
    # 2. 翻译功能
    print("翻译功能测试：")
    chinese_text = "今天天气很好，我们去公园散步吧。"
    english_translation = ai_assistant.translate(chinese_text, '英文')
    print(f"原文：{chinese_text}\n")
    print(f"译文：{english_translation}\n")

def run_code_helper(ai_assistant):
    # 3. 代码助手
    print("代码助手测试：")
    code_task = "写一个函数，计算列表中所有偶数的和"
    generated_code = ai_assistant.code_helper(code_task)
    print(f"需求：{code_task}\n")
    print(f"生成的代码：\n{generated_code}\n")

def run_creative_writing(ai_assistant):
    # 4. 创意写作
    print("创意写作测试：")
    creative_topic = "程序员的日常生活"
    creative_content = ai_assistant.creative_writing(creative_topic, '幽默')
    print(f"主题：{creative_topic}\n")
    print(f"创意内容：{creative_content}\n")

def main():
    # 创建AI助手实例
    my_ai = AIAssistant(client)
    print("你的专属AI助手已就绪！")
    run_basic_chat(my_ai)
    run_translate(my_ai)
    run_code_helper(my_ai)
    run_creative_writing(my_ai)

if __name__ == '__main__':
    main()
