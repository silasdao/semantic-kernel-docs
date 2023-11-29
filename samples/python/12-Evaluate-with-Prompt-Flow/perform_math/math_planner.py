import asyncio
from promptflow import tool

import semantic_kernel as sk
from semantic_kernel.planning.sequential_planner import SequentialPlanner
from plugins.MathPlugin.Math import Math as Math
from promptflow.connections import (
    AzureOpenAIConnection,
)

from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextCompletion,
)


@tool
def my_python_tool(
    input: str,
    deployment_type: str,
    deployment_name: str,
    AzureOpenAIConnection: AzureOpenAIConnection,
) -> str:
    # Initialize the kernel
    kernel = sk.Kernel(log=sk.NullLogger())

    # Add the chat service
    if deployment_type == "chat-completion":
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(
                deployment_name,
                AzureOpenAIConnection.api_base,
                AzureOpenAIConnection.api_key,
            ),
        )
    elif deployment_type == "text-completion":
        kernel.add_text_completion_service(
            "text_completion",
            AzureTextCompletion(
                deployment_name,
                AzureOpenAIConnection.api_base,
                AzureOpenAIConnection.api_key,
            ),
        )

    planner = SequentialPlanner(kernel=kernel)

    # Import the native functions
    math_plugin = kernel.import_skill(Math(), "MathPlugin")

    ask = f"Use the available math functions to solve this word problem: {input}"

    plan = asyncio.run(planner.create_plan_async(ask))

    # Execute the plan
    result = asyncio.run(kernel.run_async(plan)).result

    for step in plan._steps:
        print(f"Function: {step.skill_name}.{step._function.name}")
        print(f"Input vars: {str(step.parameters.variables)}")
        print(f"Output vars: {str(step._outputs)}")
    print(f"Result: {str(result)}")

    return str(result)
