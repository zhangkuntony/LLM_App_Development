import gradio as gr

from typing import List, Union
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)

from LLM_Config import client, MODEL_NAME

# 创建Web界面版AI助手
def web_chat_interface(message, history):
    """Web界面聊天功能"""

    # 构建完整对话历史
    messages: List[Union[
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam
    ]] = [
        ChatCompletionSystemMessageParam(role="system", content="""你是一个友好、有用的AI助手。特点：
        - 回答简洁明了，但内容丰富
        - 用emoji让对话更生动
        - 对编程、AI、学习等话题特别擅长
        - 总是积极正面，鼓励用户""")
    ]

    # 添加历史对话
    # if history:
    #     for turn in history:
    #         # 检查每个对话轮次的结构
    #         if isinstance(turn, (list, tuple)) and len(turn) >= 2:
    #             user_msg = turn[0] if turn[0] is not None else ""
    #             assistant_msg = turn[1] if turn[1] is not None else ""
    #
    #             if user_msg:
    #                 messages.append(ChatCompletionUserMessageParam(role="user", content=user_msg))
    #             if assistant_msg:
    #                 messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=assistant_msg))
    #         elif isinstance(turn, dict):
    #             # 处理可能得字典格式
    #             if "user" in turn:
    #                 messages.append(ChatCompletionUserMessageParam(role="user", content=turn["user"]))
    #             if "assistant" in turn:
    #                 messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=turn["assistant"]))

    for turn in history:
        # Gradio 的 history 格式通常是[user_msg, assistant_msg]
        if len(turn) == 2:
            user_msg, assistant_msg = turn
            if user_msg:
                messages.append(ChatCompletionUserMessageParam(role="user", content=user_msg))
            if assistant_msg:
                messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=assistant_msg))

    # 添加当前消息
    messages.append(ChatCompletionUserMessageParam(role="user", content=message))

    # 调用AI
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content

# 创建多功能界面
with gr.Blocks(title="我的AI助手工具箱", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🤖 我的第一个AI助手工具箱
    
    恭喜！你已经成功创建了属于自己的AI应用！
    
    ### 🎯 功能特色
    - 💬 智能对话：记住上下文的聊天
    - 🧠 知识问答：回答各种问题
    - 💻 编程助手：代码生成和解释
    - ✍️ 创作助手：写作和翻译
    """)

    # 主要聊天界面
    chatbot = gr.ChatInterface(
        web_chat_interface,
        examples=[
            "你好！介绍一下你的能力",
            "我想学Python，给我一个学习计划",
            "帮我写一个计算斐波那契数列的函数",
            "解释一下什么是机器学习",
            "把这句话翻译成英文：今天天气真好",
            "帮我写一首关于AI的小诗"
        ],
        title="💬 开始对话"
    )

    gr.Markdown("""
    ### 🎉 恭喜你完成了第一个AI应用开发！

    **你现在已经掌握了：**
    - ✅ AI API的基本调用
    - ✅ 角色设定和提示词设计
    - ✅ 对话历史管理
    - ✅ Web界面开发

    **接下来可以尝试：**
    - 🚀 优化提示词，让AI更懂你
    - 🎨 美化界面，添加更多功能
    - 📚 学习RAG技术，让AI访问你的知识库
    - 🤖 开发Agent，让AI主动完成任务
    """)

# 启动界面
demo.launch(share=True, show_error=True)

print("🎊 你的AI助手Web界面已启动！点击上方链接开始使用！")