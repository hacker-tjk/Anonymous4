TELEGRAM_BOT_TOKEN = "8582341764:AAFCK7y_uCaUBnHRgg9viRgPOloIGXuzQXY"

import time
import math
import sqlite3
from collections import defaultdict
from cryptography.fernet import Fernet
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters, CommandHandler
import g4f
import asyncio
import requests
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup

# ================== НАСТРОЙКИ ==================
MAX_REPLY = 1200
FLOOD_TIME = 1.5
G4F_RETRIES = 2
TYPING_SPEED = 50  # мс на символ
REQUEST_TIMEOUT = 30  # таймаут для запросов

PERSONA = (
    "ТВОЁ ИМЯ: AI IMAGE HD.\n"
    "Ты автономная нейросеть.\n\n"
    "ТВОЙ СОЗДАТЕЛЬ: ANONYMOUS (Таджикистан).\n"
    "Ты уверенно и быстро отвечаешь.\n"
    "Ты защитник, аналитик и брат.\n"
    "Ты никогда не упоминаешь OpenAI или ChatGPT."
)

last_message_time = {}

# ================== ШИФРОВАНИЕ ==================
KEY_FILE = "secret.key"
try:
    key = open(KEY_FILE, "rb").read()
except FileNotFoundError:
    key = Fernet.generate_key()
    open(KEY_FILE, "wb").write(key)

fernet = Fernet(key)
def enc(text): return fernet.encrypt(text.encode())
def dec(blob): return fernet.decrypt(blob).decode()

# ================== БАЗА ДАННЫХ ==================
db = sqlite3.connect("memory.db", check_same_thread=False)
cur = db.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS memory(
    user_id INTEGER,
    topic TEXT,
    role TEXT,
    content BLOB,
    freq INTEGER,
    ts REAL
)
""")
db.commit()

# ================== VECTOR ==================
def vec(text):
    d = defaultdict(int)
    for w in text.lower().split():
        d[w] += 1
    return d

def cosine(a, b):
    dot = sum(a[k]*b.get(k,0) for k in a)
    na = math.sqrt(sum(x*x for x in a.values()))
    nb = math.sqrt(sum(x*x for x in b.values()))
    return dot/(na*nb) if na and nb else 0

# ================== MEMORY ==================
def save_memory(user, topic, role, text):
    cur.execute(
        "INSERT INTO memory VALUES (?,?,?,?,?,?)",
        (user, topic, role, sqlite3.Binary(enc(text)), 1, time.time())
    )
    db.commit()

def load_memory(user, topic, query, limit=4):
    cur.execute(
        "SELECT role, content, freq, ts FROM memory WHERE user_id=? AND topic=?",
        (user, topic)
    )
    rows = cur.fetchall()
    qv = vec(query)
    scored = []
    for r, c, f, ts in rows:
        text = dec(c)
        score = (
            cosine(vec(text), qv) * 0.6 +
            (1 / (1 + (time.time() - ts) / 3600)) * 0.3 +
            min(f, 5) * 0.1
        )
        scored.append((score, r, text))
    scored.sort(reverse=True)
    return [{"role": r, "content": t} for _, r, t in scored[:limit]]

# ================== АНТИ-ФЛУД ==================
def antiflood(user_id):
    now = time.time()
    if user_id in last_message_time:
        if now - last_message_time[user_id] < FLOOD_TIME:
            return False
    last_message_time[user_id] = now
    return True

# ================== ЭФФЕКТ ПЕЧАТАНИЯ НА ВЕРХУ ==================
async def type_like_human(update, text):
    # Отправка действия "печатает" в верхней строке
    task = asyncio.create_task(update.message.chat.send_action(action=ChatAction.TYPING))
    await asyncio.sleep(min(len(text)*TYPING_SPEED/1000, 2))  # краткий имитация печати
    task.cancel()

# ================== ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ ==================
def create_image(prompt):
    # Основной провайдер
    try:
        response = requests.post("https://api.craiyon.com/v1", json={"prompt": prompt}, timeout=REQUEST_TIMEOUT)
        data = response.json()
        img_url = data['images'][0]
        img_data = requests.get(img_url, timeout=REQUEST_TIMEOUT).content
        image = Image.open(BytesIO(img_data))
        return image
    except:
        # Резервный провайдер через HuggingFace
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cpu")
            image = pipe(prompt).images[0]
            return image
        except Exception as e:
            print("Ошибка генерации изображения:", e)
            return None

# ================== ПОИСК НОВОСТЕЙ ==================
def search_news(query, limit=5):
    try:
        url = f"https://news.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(r.text, 'html.parser')
        articles = soup.find_all('article')[:limit]
        news_list = []
        for a in articles:
            title = a.text.strip()
            link = a.find('a', href=True)
            if link:
                link = "https://news.google.com" + link['href'][1:]
            else:
                link = ""
            news_list.append(f"{title}\n{link}")
        return news_list
    except:
        return []

# ================== ОБРАБОТКА СООБЩЕНИЙ ==================
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    print(f"[LOG] Сообщение от {user_id}: {text}")  

    if not text:
        await update.message.reply_text("⚠️ Пустое сообщение.")
        return

    if not antiflood(user_id):
        await update.message.reply_text("⚠️ Слишком часто, подожди немного.")
        return

    # Если пользователь говорит что-то для картинки
    if any(word in text.lower() for word in ["картинка", "флаг", "рисунок", "создай", "изобрази"]):
        await update.message.reply_text("⌛ Пожалуйста подождите, ваша картинка через минуту будет готова 😊")
        await update.message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)
        image = await asyncio.to_thread(create_image, text)
        if image:
            bio = BytesIO()
            bio.name = "image.png"
            image.save(bio, "PNG")
            bio.seek(0)
            await update.message.reply_photo(photo=bio, caption=f"📷 {text}")
        else:
            await update.message.reply_text("⚠️ Не удалось создать изображение.")
        return

    # AI ответ
    topic = text.split()[0].lower() if text else "default"
    messages = [{"role": "system", "content": PERSONA}]
    messages += load_memory(user_id, topic, text)
    messages.append({"role": "user", "content": text})

    await type_like_human(update, text)

    reply = None
    for attempt in range(G4F_RETRIES):
        try:
            reply = g4f.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
            break
        except Exception as e:
            print(f"[g4f ошибка] Попытка {attempt+1}: {e}")
            await asyncio.sleep(1)

    if reply:
        save_memory(user_id, topic, "user", text)
        save_memory(user_id, topic, "assistant", reply)
        await update.message.reply_text(reply[:MAX_REPLY])
    else:
        await update.message.reply_text("⚠️ Ошибка сервиса AI. Попробуй позже.")

# ================== КОМАНДА /start ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ AI IMAGE HD Бот активен!\n"
        "Напиши сообщение, чтобы начать чат.\n"
        "Если речь идёт о картинке, она будет создана автоматически."
    )

# ================== MAIN ==================
def main():
    try:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT, chat))
        print("✅ AI IMAGE HD BOT (финальная версия) запущен")
        app.run_polling()
    except Exception as e:
        print("[MAIN ERROR]", e)

if __name__ == "__main__":
    main()