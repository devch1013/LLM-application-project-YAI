"""
작성자: 김대성
"""

import asyncio
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from llm.yainoma import YAINOMA

load_dotenv(".env_secret")

# intents 설정
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True  # 서버 정보 접근을 위해 필요


# 봇 생성
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

model = YAINOMA(model_dir="/home/elicer/LLM_project_YAI/llm/llama/llama_weights/fine_tune_200_inst")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}!")
    for guild in bot.guilds:
        for channel in guild.text_channels:
            try:
                await channel.send(
                    """
                YAIbot 출근했습니다! 명령어가 궁금하시면 !help를 보내주세요!
                """
                )
            except discord.Forbidden:
                # 채널에 메시지를 보낼 수 없는 경우 무시
                print(f"Cannot send message to {channel.name} in {guild.name}")


bot_anchor = {"AI": "[ai]", "RAG": "[rag]", "ARXIV": "[arxiv]"}


# on_message 이벤트에서 !a 명령어 처리 부분 수정
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # 봇 자신의 메시지에는 반응하지 않음

    await bot.process_commands(message)  # 명령어도 함께 처리하기 위함

    if message.content.lower().startswith(bot_anchor["AI"]):
        prompt = message.content[
            len(bot_anchor["AI"]) :
        ].strip()  # '[AI]' 이후의 텍스트를 프롬프트로 사용
        response = model.simple_qa(prompt)
        await message.channel.send(response)

    elif message.content.lower().startswith(bot_anchor["RAG"]):
        prompt = message.content[
            len(bot_anchor["RAG"]) :
        ].strip()  # '[AI]' 이후의 텍스트를 프롬프트로 사용
        response = model.paper_RAG(prompt)
        await message.channel.send(response)

    elif message.content.lower().startswith(bot_anchor["ARXIV"]):
        prompt = message.content[
            len(bot_anchor["ARXIV"]) :
        ].strip()  # '[AI]' 이후의 텍스트를 프롬프트로 사용
        response = model.general_RAG(prompt)
        await message.channel.send(response)

    # elif message.content.lower().startswith('!a'):
    #     prompt = message.content[2:].strip()  # '!a' 이후의 텍스트를 프롬프트로 사용
    #     response = await send_request_to_api(prompt)  # FastAPI 서버에 요청을 보내고 응답을 받음
    #     await message.channel.send(response)  # 받은 응답을 Discord 채널에 전송

    else:
        greeting_keywords = ["hello", "hi", "안녕하세요", "안녕", "하이"]
        if any(keyword in message.content.lower() for keyword in greeting_keywords):
            await message.channel.send(f"안녕하세요, {message.author.display_name}님!")


@bot.command()
async def help(ctx):
    help_response = """
[ai]를 앞에 붙이면 YAIbot이 응답합니다.
[rag]를 앞에 붙이면 AI, YAI에 관련된 문서를 통한 RAG 응답이 생성됩니다.
[arxiv]를 앞에 붙이면 YAIbot이 논문을 검색해 응답을 생성합니다. 
    """
    await ctx.send(help_response)


@bot.command(name="shutdown")
@commands.is_owner()  # 봇 소유자만 사용할 수 있는 명령어
async def shutdown(ctx):
    await ctx.send("야이 봇 이제 퇴근합니다. 안녕")
    await bot.close()


# 봇 토큰
TOKEN = os.getenv("DISCORD_TOKEN")

try:
    bot.run(TOKEN)
except KeyboardInterrupt:
    # 키보드 인터럽트 (Ctrl+C)로 봇을 종료할 때
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.close())
finally:
    print("봇이 종료되었습니다.")
