import json

from datasets import load_dataset
from mistral_common.protocol.instruct.messages import (
    UserMessage,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    ToolCall,
    FunctionCall,
    AssistantMessage,
    Function,
    Tool,
)


def run(dataset: str, model: str):
    tokenizer = MistralTokenizer.from_model(model)

    data = load_dataset(dataset, split="train").map(
        lambda x: {"text": tokenize(row=x, encoder=tokenizer)}
    )
    data = data.train_test_split(train_size=0.001)

    data["train"].to_json("./train.jsonl")
    data["test"].to_json("./test.jsonl")


def tokenize(row: dict, encoder: MistralTokenizer):
    completion = encoder.encode_chat_completion(
        ChatCompletionRequest(
            tools=[
                Tool(
                    function=Function(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    )
                )
                for t in json.loads(row["tools"])
            ],
            messages=[
                UserMessage(content=row["query"]),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="VvvODy9mT",
                            function=FunctionCall(
                                name=t["name"],
                                arguments=t["arguments"],
                            ),
                        )
                        for t in json.loads(row["answers"])
                    ],
                ),
            ],
            model="ppt-0.0.1",
        )
    )
    return completion.text


if __name__ == "__main__":
    run("Salesforce/xlam-function-calling-60k")
