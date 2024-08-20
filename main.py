"""
작성자: 김대성
"""

import asyncio
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from llm.yainoma import YAINOMA
from llm.momugzi import restaurant_bot

load_dotenv(".env_secret")

# intents 설정
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True  # 서버 정보 접근을 위해 필요


# 봇 생성
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

model = YAINOMA(model_dir="/home/elicer/LLM_project_YAI/llm/llama/llama_weights/fine_tune_200_inst")
default_model = YAINOMA()


# 버튼 클래스 구성

class RestaurantSelection(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.lunch = False
        self.dinner = False
        self.alcohol = False
        self.cafe = False

    # send_response 메서드에서 문제를 확인하기 위한 디버그 코드 추가
    async def send_response(self, interaction):
        try:
            # 버튼 선택이 완료된 후 restaurant_bot 함수 호출
            response = restaurant_bot(self.lunch, self.dinner, self.alcohol, self.cafe)
            if response:  # response가 존재하는지 확인
                await interaction.response.send_message(response)
            else:
                await interaction.response.send_message("죄송합니다, 메뉴를 추천할 수 없습니다.")
        except Exception as e:
            # 예외 발생 시 에러 메시지 출력
            await interaction.response.send_message(f"오류가 발생했습니다: {str(e)}")

    @discord.ui.button(label="점심밥", style=discord.ButtonStyle.primary)
    async def lunch_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.lunch = True
        await interaction.response.send_message("점심밥 선택됨!", ephemeral=True)

    @discord.ui.button(label="저녁밥", style=discord.ButtonStyle.primary)
    async def dinner_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.dinner = True
        await interaction.response.send_message("저녁밥 선택됨!", ephemeral=True)

    @discord.ui.button(label="술", style=discord.ButtonStyle.primary)
    async def alcohol_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.alcohol = True
        await interaction.response.send_message("술 선택됨!", ephemeral=True)

    @discord.ui.button(label="카페", style=discord.ButtonStyle.primary)
    async def cafe_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.cafe = True
        await interaction.response.send_message("카페 선택됨!", ephemeral=True)

        # done_button의 경우에서 send_response 호출을 확실히 하도록 수정
    @discord.ui.button(label="선택 완료", style=discord.ButtonStyle.success)
    async def done_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_response(interaction)
        self.stop()  # 모든 선택이 완료되면 View를 종료

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

@bot.event
async def on_close():
    print("Bot closed")
    for guild in bot.guilds:
        for channel in guild.text_channels:
            try:
                await channel.send(
                    """
                YAIbot 퇴근합니다!
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

    print("MESSAGE: ", message)
    # 특정 명령어 체크
    if message.content == '!뭐먹지':
        view = RestaurantSelection()
        await message.channel.send("메뉴를 선택하세요: \n(선택 완료를 눌러야 응답이 출력됩니다.)", view=view)
        return

    await bot.process_commands(message)  # 명령어도 함께 처리하기 위함

    if message.content.lower().startswith(bot_anchor["AI"]):
        print("========AI=========")
        prompt = message.content[
            len(bot_anchor["AI"]) :
        ].strip()  # '[AI]' 이후의 텍스트를 프롬프트로 사용
        response = model.simple_qa(prompt)

    elif message.content.lower().startswith(bot_anchor["RAG"]):
        print("========RAG=========")
        prompt = message.content[
            len(bot_anchor["RAG"]) :
        ].strip() 
        response = default_model.general_RAG(prompt)

    elif message.content.lower().startswith(bot_anchor["ARXIV"]):
        print("========ARXIV=========")
        prompt = message.content[
            len(bot_anchor["ARXIV"]) :
        ].strip() 
        response, doc = default_model.paper_RAG(prompt)
        response = response + "\n참고 논문: " + doc["title"] + "\n논문 링크: " + doc["url"]

    else:
        greeting_keywords = ["hello", "hi", "안녕하세요", "안녕", "하이"]
        if any(keyword in message.content.lower() for keyword in greeting_keywords):
            await message.channel.send(f"안녕하세요, {message.author.display_name}님!")
            return
        else:
            return
    print("ANSWER: ", response)
    if response is None:
        await message.channel.send("Inference가 진행중입니다. 잠시 후 시도해주세요.")
        return
    await message.channel.send(response)


@bot.command()
async def help(ctx):
    help_response = """
[ai]        메시지 앞에 붙이면 YAIbot이 응답합니다.
[rag]       메시지 앞에 붙이면 AI, YAI에 관련된 문서를 통한 RAG 응답이 생성됩니다.
[arxiv]     메시지 앞에 붙이면 YAIbot이 논문을 검색해 응답을 생성합니다.

!yai        야이에 대한 기본 정보를 알려줍니다.   
    """
    await ctx.send(help_response)


@bot.command()
async def yai(ctx):
    response = f"""
    {ctx.author.mention} **YAI 동아리에 오신 것을 환영합니다!**
모두 친절하게 대해주시고, 도움이 필요하시면 언제든지 운영진에게 문의해 주세요.

- **노션 사이트**: <https://y-ai.notion.site/Yonsei-Artificial-Intelligence-YAI-23fc16b649b64aa7bd0e2b6c1a68cd9d?pvs=74>
- **GitHub**: <https://github.com/yonsei-YAI/yonsei-YAI>
- **YouTube**: <https://www.youtube.com/@YonseiAI>
- **LinkedIn**: <https://www.linkedin.com/in/yai-yonsei-578022238/?originalSubdomain=kr>
- **블로그**: <https://yai-yonsei.tistory.com/>

해당 링크들을 통해 YAI의 다양한 활동과 정보를 확인하실 수 있습니다!

!뭐먹지 를 입력해서 메뉴를 추천받아보세요!
"""
    await ctx.send(response)


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
