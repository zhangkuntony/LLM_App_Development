# 实际应用：智能参数选择器
from openai.types.chat import ChatCompletionUserMessageParam
from LLM_Config import client
from config import MODEL_NAME


class SmartAIAssistant:
    """智能AI助手，根据任务类型自动选择最佳参数"""

    def __init__(self, smart_ai_client):
        self.client = smart_ai_client

        # 预设的参数配置（基于官方temperature范围0-2）
        self.task_configs = {
            "code": { "temperature": 0.1, "top_p": 0.3, "max_tokens": 500 },
            "math": { "temperature": 0, "top_p": 0.1, "max_tokens": 200 },
            "translate": {"temperature": 0.2, "top_p": 0.4, "max_tokens": 300},
            "creative": {"temperature": 1.2, "top_p": 0.9, "max_tokens": 400},
            "brainstorm": {"temperature": 1.5, "top_p": 0.95, "max_tokens": 400},
            "chat": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 300},
            "summary": {"temperature": 0.3, "top_p": 0.6, "max_tokens": 250}
        }

    def smart_completion(self, prompt,  task_type="chat", custom_params=None):
        """根据任务类型智能选择参数"""

        # 获取预设配置
        config = self.task_configs.get(task_type, self.task_configs["chat"])

        # 如果有自定义参数，则覆盖默认值
        if custom_params:
            config.update(custom_params)

        print(f"任务类型：{task_type}")
        print(f"使用参数：Temperature={config['temperature']}, TopP={config['top_p']}, MaxTokens={config['max_tokens']}")
        print(f"用户输入：{prompt}")
        print("-" * 60)

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    ChatCompletionUserMessageParam(role="user", content=prompt)
                ],
                temperature=config['temperature'],
                top_p=config['top_p'],
                max_tokens=config['max_tokens']
            )

            result = response.choices[0].message.content
            print(f"AI回复：{result}\n")
            return result

        except Exception as e:
            print(f"错误：{e}\n")
            return None


def main():
    # 创建智能助手实例
    smart_ai = SmartAIAssistant(client)

    # 测试不同任务类型
    print("代码生成任务测试：")
    smart_ai.smart_completion(
        "写一个Python函数，实现二分查找算法",
        task_type="code"
    )

    print("\n" + "=" * 80)
    print("数学计算任务测试：")
    smart_ai.smart_completion(
        "计算复利：本金10000元，年利率5%，10年后的本息合计是多少？",
        task_type="math"
    )

    print("\n" + "=" * 80)
    print("🎨 创意写作任务测试：")
    smart_ai.smart_completion(
        "写一个关于时间旅行者的有趣故事开头",
        task_type="creative"
    )

    print("\n" + "=" * 80)
    print("🧠 头脑风暴任务测试（高temperature=1.5）：")
    smart_ai.smart_completion(
        "为'环保主题咖啡店'想5个创新的商业点子",
        task_type="brainstorm"
    )

if __name__ == "__main__":
    main()
