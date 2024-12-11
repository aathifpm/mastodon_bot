import asyncio
from mastodon import Mastodon
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import time
from datetime import datetime, timedelta
import requests
from io import BytesIO
from PIL import Image
import json
import random
import logging
from aiohttp import web

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class PostStyle:
    MEME = "meme"
    ENTERTAINER = "entertainer"
    INFORMATIVE = "informative"
    STORYTELLER = "storyteller"
    ANALYST = "analyst"

class MastodonBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize credentials
        self.credentials = {
            'instance_url': os.getenv('MASTODON_INSTANCE_URL'),
            'client_id': os.getenv('MASTODON_CLIENT_ID'),
            'client_secret': os.getenv('MASTODON_CLIENT_SECRET'),
            'access_token': os.getenv('MASTODON_ACCESS_TOKEN'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY')
        }
        
        # Initialize Mastodon client
        self.client = Mastodon(
            client_id=self.credentials['client_id'],
            client_secret=self.credentials['client_secret'],
            access_token=self.credentials['access_token'],
            api_base_url=self.credentials['instance_url']
        )
        
        # Initialize Gemini
        genai.configure(api_key=self.credentials['gemini_api_key'])
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Rate limiting settings
        self.last_request_time = 0
        self.request_count = 0
        self.max_requests_per_minute = 30
        self.retry_delay = 5
        
        # Auto-posting settings
        self.auto_post_interval = 1800  # 30 minutes
        self.last_post_time = time.time()
        self.post_count = 0
        self.max_daily_posts = 48  # 2 posts per hour
        self.current_style = PostStyle.ENTERTAINER
        
        # DM handling
        self.replied_dms = set()
        self.dm_context_file = "dm_context.json"
        self._load_dm_context()
        
        # Configuration
        self.post_config = {
            "use_hashtags": True,
            "max_length": 240,
            "blacklisted_words": []
        }
        
        # Monitoring settings
        self.hashtags_to_monitor = ["tech", "AI", "programming"]  # Customize these
        self.is_running = False 

    async def _handle_rate_limit(self):
        """Handle API rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # Reset counter after a minute
        if time_diff >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If we're at the limit, wait
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - time_diff
            if wait_time > 0:
                print(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        # Add small delay between requests
        await asyncio.sleep(2)  # Add 2-second delay between requests
        self.request_count += 1

    async def generate_entertainment_response(self, post_text: str, status: Dict = None, max_retries=3) -> str:
        """Generate a short, fun response using Gemini, including image analysis if present"""
        clean_text = self._clean_html(post_text)
        
        # Get media attachments if status is provided
        images = []
        if status:
            media_attachments = self._get_media_attachments(status)
            for media in media_attachments:
                if image := await self._download_image(media['url']):
                    images.append({
                        'image': image,
                        'description': media['description']
                    })

        # Modify prompt based on presence of images
        base_prompt = f"""Create a fun, short response to this post: "{clean_text}" """
        
        if images:
            base_prompt += "\nThe post includes images which I'll analyze for context."
            base_prompt += "\nIncorporate relevant details from the images in the response."
        
        prompt = base_prompt + """
        Rules:
        - Maximum 2 sentences
        - Include 1-2 emojis
        - Be witty and friendly
        - Match the post's tone
        - Add a relevant pop culture reference if it fits naturally
        - Reference image content naturally (if images present)
        
        Format: Just the response text with emojis.
        """
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(5)  # Wait 5 seconds between attempts
                
                if images:
                    # Use multimodal generation if images are present
                    generation_config = {
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40
                    }
                    
                    # Create a list of content parts for multimodal input
                    content_parts = [prompt]
                    for img_data in images:
                        content_parts.append(img_data['image'])
                        if img_data['description']:
                            content_parts.append(f"Image description: {img_data['description']}")
                    
                    response = self.model.generate_content(
                        content_parts,
                        generation_config=generation_config
                    )
                else:
                    # Text-only generation
                    response = self.model.generate_content(prompt)
                
                return response.text[:240].strip()  # Maintain character limit
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error generating response: {str(e)}")
                    return "✨ Interesting perspective! Thanks for sharing! 🌟"
    async def schedule_auto_posts(self):
        """Main loop for scheduled auto-posting and DM handling"""
        print("Starting scheduled auto-posting service...")
        first_run = True
        while True:
            try:
                current_time = time.time()
                
                # Handle regular posts
                if first_run or current_time - self.last_post_time >= self.auto_post_interval:
                    if self.post_count < self.max_daily_posts:
                        await self.create_scheduled_post()
                        self.last_post_time = current_time
                        self.post_count += 1
                        print(f"Auto-post complete. Posts today: {self.post_count}/{self.max_daily_posts}")
                    first_run = False
                
                # Handle DMs if enabled
                if self.dm_settings["enabled"] and self.dm_settings["auto_reply"]:
                    if current_time % self.dm_settings["reply_interval"] < 60:
                        await self.handle_direct_messages()
                
                # Handle auto-likes
                if current_time % 300 < 60:  # Check every 5 minutes
                    await self.auto_like_trending_posts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in auto-posting loop: {str(e)}")
                await asyncio.sleep(300)
    async def run_forever(self):
        """Main loop for running the bot continuously"""
        self.is_running = True
        print("Starting Mastodon bot...")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Handle regular posts
                if current_time - self.last_post_time >= self.auto_post_interval:
                    if self.post_count < self.max_daily_posts:
                        await self.create_scheduled_post()
                        self.last_post_time = current_time
                        self.post_count += 1
                        print(f"Auto-post complete. Posts today: {self.post_count}/{self.max_daily_posts}")
                
                # Monitor hashtags
                for hashtag in self.hashtags_to_monitor:
                    await self.monitor_hashtag(hashtag)
                
                # Handle DMs
                await self.handle_direct_messages()
                
                # Reset post count at midnight
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.post_count = 0
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def monitor_hashtag(self, hashtag: str):
        """Monitor and respond to hashtag posts"""
        try:
            await self._handle_rate_limit()
            posts = await self.search_hashtag(hashtag, limit=5)
            
            for post in posts:
                if random.random() < 0.3:  # 30% chance to respond
                    response = await self.generate_entertainment_response(post['content'])
                    await self.reply_to_post(post['id'], response)
                    print(f"Replied to post in #{hashtag}")
                    await asyncio.sleep(30)  # Wait between responses
                    
        except Exception as e:
            print(f"Error monitoring hashtag #{hashtag}: {str(e)}")

async def health_check():
    """Simple health check endpoint"""
    return web.Response(text="OK", status=200)

async def start_health_check():
    app = web.Application()
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

if __name__ == "__main__":
    # Create and run the bot
    bot = MastodonBot()
    
    try:
        logging.info("Bot started successfully")
        asyncio.run(bot.run_forever())
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        # Attempt to restart
        os.system('docker-compose restart mastodon-bot')
    except KeyboardInterrupt:
        print("\nBot stopped by user")    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
