from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from LLM_Config import client, MODEL_NAME           # 导入配置

def first_ai_chat():
    print("准备开始第一次AI对话")

    # 向AI发送消息
    user_question = "你好！请用一句话介绍一下你自己，并告诉我你能帮我做什么？"

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            ChatCompletionUserMessageParam(role="user", content=user_question)
        ]
    )
    print(response)

    ai_reply = response.choices[0].message.content

    print(f"👤 你问：{user_question}")
    print(f"🤖 AI答：{ai_reply}")
    print("\n🎊 恭喜！你刚刚完成了第一次与AI的对话！")

def chat_poem():
    # 🎪 互动环节：问AI任何问题！

    # 💡 试试这些有趣的问题：
    # "帮我写一首关于编程的小诗"
    # "解释一下什么是人工智能，用小学生能理解的话"
    # "给我推荐3本值得读的书，并说明理由"
    # "我想学Python，给我一个学习计划"
    my_question = "帮我写一首关于编程的小诗"               # 在这里修改你的问题

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            ChatCompletionUserMessageParam(role="user", content=my_question)
        ],
        max_tokens=200
    )

    print(f"👤 你问：{my_question}")
    print(f"🤖 AI答：{response.choices[0].message.content}")
    print("\n✨ 神奇吧？这就是大模型的力量！")


# 角色设定的魔法：让AI变身不同角色
def ask_ai_with_role(user_question, ai_role):
    """向不同角色的AI提问"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            ChatCompletionSystemMessageParam(role="system", content=ai_role),               # 这里设定AI的角色
            ChatCompletionUserMessageParam(role="user", content=user_question)
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

def chat_with_different_role():
    # 同一个问题，不同角色的AI会如何回答？
    question = "什么是编程？"

    roles = [
        # 角色1：专业老师
        { "role_name": "专业老师", "role_description": "你是一位经验丰富的编程老师，用专业但易懂的方式解释概念。"},
        # 角色2：幽默朋友
        { "role_name": "幽默朋友", "role_description": "你是一个幽默风趣的朋友，喜欢用比喻和段子来解释事物。"},
        # 角色3:5岁小孩
        { "role_name": "5岁小孩", "role_description": "你是一个5岁小朋友，用最简单天真的话来回答问题。"}
    ]

    print(f"问题: {question}\n")
    for role in roles:
        role_name = role["role_name"]
        role_description = role["role_description"]
        answer = ask_ai_with_role(question, role_description)
        print(f"{role_name}说: {answer}")

    print("🎪 是不是很神奇？同一个AI，不同的角色设定，完全不同的回答风格！")

if __name__ == "__main__":
    first_ai_chat()
    # chat_poem()
    # chat_with_different_role()