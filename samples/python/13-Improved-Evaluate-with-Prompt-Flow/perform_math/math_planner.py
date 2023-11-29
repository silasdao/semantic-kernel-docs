import asyncio
from promptflow import tool

import semantic_kernel as sk
from semantic_kernel.planning.sequential_planner import SequentialPlanner
from plugins.MathPlugin.Math import Math as Math

import config.add_completion_service


@tool
def my_python_tool(input1: str) -> str:
    # Initialize the kernel
    kernel = sk.Kernel()
    # Add a text or chat completion service using either:
    # kernel.add_text_completion_service()
    # kernel.add_chat_service()
    kernel.add_completion_service()

    planner = SequentialPlanner(kernel=kernel)

    # Import the native functions
    math_plugin = kernel.import_skill(Math(), "MathPlugin")

    ask = f"Use the available math functions to solve this word problem: {input1}"
    ask += "\n\n"
    ask += "- When using the functions, do not pass any units or symbols like $; only send valid floats.\n"
    ask += "- Only use the necessary functions needed to answer the word problem.\n"

    plan = asyncio.run(planner.create_plan_async(ask))

    # Execute the plan
    result = asyncio.run(kernel.run_async(plan)).result

    for step in plan._steps:
        print(f"Function: {step.skill_name}.{step._function.name}")
        print(f"Input vars: {str(step.parameters.variables)}")
        print(f"Output vars: {str(step._outputs)}")
    print(f"Result: {str(result)}")

    return str(result)
