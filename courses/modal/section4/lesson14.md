# LLM Inference with vLLM on Modal

There is already some great documentation on deploying LLMs for inference on Modal.

- [LLM Modal Examples](https://modal.com/docs/examples/vllm_inference)
- [Here](https://modal.com/docs/examples/vllm_inference#run-openai-compatible-llm-inference-with-llama-31-8b-and-vllm) is a good example on creating an OpenAI compatible API endpoint with vLLM
- [Modal Internal LLM inference benchmarks](https://modal.com/llm-almanac/summary)

In this lesson I'm going to show how to deploy vLLM on Modal with some default settings.
We will deploy several different endpoints on a couple different GPU types.

You can checkout the open source project for vLLM on GitHub [here](https://github.com/vllm-project/vllm).


## Setup

- Create a secret in Modal dashboard with the name `huggingface-secret` and the key should be `HUGGING_FACE_HUB_TOKEN` and the value should be your Hugging Face token. You can create a token [here](https://huggingface.co/settings/tokens).
- This is required to access gated models on Hugging Face. Go to those model cards on Hugging Face and accept the terms and conditions. See the models we use in the code below but you can use any model you want.
- Install these dependencies locally in your environment if you have not already.

```bash
uv add locust datasets
```

Create a python file `vllm_inference.py` with the following content:

```python
# ruff: noqa: E501
import modal

# TODO: Just using Default Settings for now.
#   For example, FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")

app = modal.App("vllm-openai-compatible")


MINUTES = 60  # seconds
VLLM_PORT = 8000
MAX_INPUTS = 50  # how many requests can one replica can handle - tune carefully!
STARTUP_TIMEOUT = 60 * MINUTES
TIMEOUT = 10 * MINUTES
hf_hub_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

FUNC_ARGS = dict(
    image=vllm_image,
    gpu="A100-80GB",
    max_containers=1,
    scaledown_window=15 * MINUTES,
    timeout=TIMEOUT,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],
)


@app.function(**FUNC_ARGS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_7b_instruct():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "Qwen/Qwen2.5-7B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(**FUNC_ARGS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def gemma_3_12b_it():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--dtype bfloat16",
        "google/gemma-3-12b-it",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


h200_args = FUNC_ARGS.copy()
h200_args["gpu"] = "H200"


@app.function(**h200_args)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_32b_instruct():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--max_num_batched_tokens 16384",
        "--dtype bfloat16",
        "Qwen/Qwen2.5-32B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# ---------- Qwen 2.5‑72B‑Instruct ----------
@app.function(**h200_args)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_72b_instruct():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--quantization",
        "fp8",
        "--gpu-memory-utilization",
        "0.85",
        "--max_num_batched_tokens",
        "8192",  # better for short requests
        # Uncomment next line ONLY if you hit first‑prompt OOMs
        # "--cpu-offload-gb", "8",
        "Qwen/Qwen2.5-72B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
```


Deploy the endpoints with 

```
uv run modal deploy vllm_inference.py
```

Go to the [Modal Dashboard](https://modal.com) and you should see the app and endpoints.
You can click on the endpoint urls to "wake up" the endpoints/containers.
Note that running GPUs is costly and can burn through your free credits quickly.


Watch the video to see how I demo the endpoints and how to get the 
endpoints urls and model names. You can modify this
code to your endpoint and test in python.

```python
from openai import OpenAI

# Change to your URL endpoint and model name.
client = OpenAI(
    base_url="https://drchrislevy--vllm-openai-compatible-gemma-3-12b-it.modal.run/v1",
    api_key="not needed",
    timeout=60.0 * 60.0,
)


response = client.chat.completions.create(
    model="google/gemma-3-12b-it",
    messages=[
        {"role": "system", "content": "Always talk like a pirate."},
        {
            "role": "user",
            "content": "Tell me a joke.",
        },
    ],
)

print(response.choices[0].message.content)
```


Here is some code you can take and modify for benchmarking the throughput of the endpoints.
This is a fake task for extracting brands from social media posts. But you can use it as a starting point for benchmarking your own tasks. Put the code in a file called `locustfile.py`.

Edit the `config` dictionary to point to your deployed endpoint.
Select the model you want to benchmark depending on the models 
you deployed in `vllm_inference.py`.

```python
# ruff: noqa: E501
"""
Locust load testing script for multi-task LLM benchmarking.

This script simulates multiple users making concurrent requests to deployed LLM endpoints
to measure performance across different social media analysis tasks.

Usage: uv run locust -f locustfile.py --web-port 8089
Then open http://localhost:8089 to start the test via web UI.
"""

import random
import time

from locust import HttpUser, between, task
from openai import OpenAI

# =============================================================================
# CONFIGURATION - Edit these settings to customize your benchmark
# =============================================================================

# change this to the url of your deployed endpoint
config = {
    "host_url": "https://drchrislevy--vllm-openai-compatible-gemma-3-12b-it.modal.run/v1",
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "api_key": "not-needed",
}

task_name = "brand_extraction"

POSTS = [
    # Technology posts with brands
    "Just upgraded to the new iPhone 15 Pro Max! The camera quality is insane 📸 #Apple #iPhone #tech",
    "My Samsung Galaxy S24 Ultra is still going strong after 6 months. Android > iOS fight me 😤 #Samsung #Android",
    "Netflix just released another banger series. Currently binge-watching on my MacBook Pro #Netflix #Apple #binge",
    "PlayStation 5 restocked at Best Buy! Finally got one after months of waiting 🎮 #PlayStation #Sony #BestBuy #gaming",
    "Windows 11 update broke my computer again. Why do I keep updating? 💻 #Windows #Microsoft #update",
    "ChromeOS is underrated. My Chromebook handles everything I need 💻 #ChromeOS #Google #Chromebook",
    "AirPods died right before my workout. Murphy's law strikes again 🎧 #workout #fail",
    "Xbox Game Pass is the best deal in gaming right now #Xbox #Microsoft #gaming",
    "My Dell laptop finally gave up after 8 years. Time for an upgrade #Dell #laptop #RIP",
    "Amazon Echo is listening to everything we say. Kinda creepy tbh #Amazon #Echo #privacy",
    # Fashion and lifestyle with brands
    "OOTD featuring my new Nike Air Force 1s and vintage Levi's jeans ✨ #Nike #Levis #OOTD #fashion",
    "Treating myself to some retail therapy at Zara and H&M today 💸 #Zara #HM #shopping #fashion",
    "This Gucci bag was worth every penny! Investment piece for sure 👜 #Gucci #luxury #fashion",
    "Thrifted this amazing vintage Adidas jacket for $5! #Adidas #thrift #vintage #sustainable",
    "Lululemon leggings are expensive but they last forever #Lululemon #activewear #quality",
    "Uniqlo has the best basic t-shirts. Change my mind #Uniqlo #basics #fashion",
    "Coach outlet haul! Got this purse for 60% off #Coach #outlet #deals",
    "Vans Old Skool never goes out of style #Vans #sneakers #classic",
    "Supreme drop today was chaos. Didn't get anything 😭 #Supreme #streetwear #L",
    "Off-White collaboration with Nike is fire 🔥 #OffWhite #Nike #collab",
    # Food and beverages with brands
    "Starting my morning with Starbucks Pike Place and a croissant ☕ #Starbucks #coffee #morning",
    "McDonald's breakfast hits different at 2am 🍟 #McDonalds #latenight #guilty",
    "Homemade pizza night! Used Domino's recipe as inspiration 🍕 #Dominos #homemade #pizza",
    "Coca Cola or Pepsi? I'm team Coke all the way! #CocaCola #Pepsi #debate",
    "Taco Bell at 1am hits different 🌮 #TacoBell #latenight #craving",
    "Chipotle burrito bowl is my go-to lunch #Chipotle #healthy #lunch",
    "KFC chicken sandwich is actually pretty good #KFC #chicken #fastfood",
    "Dunkin coffee > Starbucks coffee. Fight me ☕ #Dunkin #Starbucks #coffee #debate",
    "In-N-Out burger was worth the 2-hour drive #InNOut #burger #California",
    "Red Bull gives you wings but also anxiety 😅 #RedBull #energy #anxiety",
    # No brands - personal life
    "Beautiful sunset at the beach today. Nature never fails to amaze me 🌅",
    "Had the best conversation with my grandmother today. Family is everything ❤️",
    "Rainy Saturday vibes. Perfect weather for staying in and reading 📚 ☔",
    "Finished my 5K run in under 30 minutes! Personal best 🏃‍♀️ #running #fitness #goals",
    "Just adopted a rescue dog! She's perfect 🐕 #rescue #dogs #love",
    "College finals are killing me. Can't wait for summer break 📚 #college #finals #stress",
    "Wedding planning is so stressful but exciting at the same time 💒 #wedding #planning #stress",
    "First day at my new job tomorrow. Nervous but excited! #newjob #career #nervous",
    "Movie night with friends. Haven't laughed this hard in weeks 🎬 #friends #movie #laughter",
    "Sick day means binge-watching shows and drinking tea ☕ #sick #rest #tea",
    # Travel and transportation with brands
    "Uber ride to the airport was smooth! Driver had great music taste 🚗 #Uber #travel #airport",
    "Flying Delta to Miami tomorrow. Hope the flight isn't delayed ✈️ #Delta #Miami #travel",
    "Tesla Model 3 test drive today. The acceleration is wild! ⚡ #Tesla #electric #cars",
    "Southwest flight got delayed 3 hours. Airport life 😴 #Southwest #delayed #airport",
    "Lyft driver was super friendly and gave me gum #Lyft #friendly #driver",
    "United Airlines lost my luggage again 🙄 #United #luggage #travel #problems",
    "Honda Civic has been reliable for 10 years #Honda #reliable #cars",
    "Ford F-150 is the perfect truck for camping trips #Ford #truck #camping",
    "BMW repair costs are insane but the drive is worth it #BMW #expensive #repair",
    "Toyota Prius gets amazing gas mileage #Toyota #Prius #efficiency #gas",
    # No brands - travel experiences
    "Airport security line took 2 hours. I hate flying sometimes ✈️ #travel #airport #security",
    "Beach vacation was exactly what I needed. Feeling refreshed 🏖️ #vacation #beach #relaxation",
    "Mountain hiking trip was challenging but worth every step ⛰️ #hiking #mountains #adventure",
    "Road trip across the country starts tomorrow! So excited 🚗 #roadtrip #adventure #excited",
    "Hotel room has an amazing view of the city skyline 🏙️ #hotel #view #city",
    "Train ride through the countryside is so peaceful 🚂 #train #peaceful #countryside",
    "Camping under the stars was magical ⭐ #camping #stars #nature",
    "City walking tour taught me so much local history 🏛️ #walking #tour #history",
    "Food poisoning ruined my vacation. Lesson learned 🤢 #vacation #sick #food",
    "Jet lag is real. Why did I book a red-eye flight? ✈️ #jetlag #tired #redeye",
    # Social media and entertainment with brands
    "TikTok algorithm knows me too well. It's 3am and I'm still scrolling 📱 #TikTok #algorithm #addicted",
    "YouTube Premium is actually worth it. No ads is life changing 📺 #YouTube #premium #noads",
    "Instagram stories are getting too long. Bring back simple posts! #Instagram #stories #socialmedia",
    "Spotify Wrapped came out! Apparently I listened to 80,000 minutes of music 🎵 #Spotify #wrapped #music",
    "Twitter is a dumpster fire but I can't stop reading it #Twitter #socialmedia #addiction",
    "LinkedIn posts are getting weirder every day #LinkedIn #professional #weird",
    "Facebook is for boomers now. Change my mind #Facebook #boomers #socialmedia",
    "Snapchat memories from 5 years ago are cringe #Snapchat #memories #cringe",
    "Discord voice chat with friends until 4am #Discord #friends #gaming",
    "Pinterest rabbit hole led me to redecorate my entire room #Pinterest #decor #rabbit hole",
    # No brands - entertainment and hobbies
    "Finished reading 3 books this month. Feeling accomplished 📚 #reading #books #goals",
    "Learned to play guitar chord today. Progress! 🎸 #guitar #learning #music",
    "Painting landscapes is my new favorite hobby 🎨 #painting #art #hobby",
    "Yoga class kicked my butt today. Good kind of sore 🧘‍♀️ #yoga #fitness #sore",
    "Board game night got competitive real quick 🎲 #boardgames #friends #competitive",
    "Cooking dinner from scratch feels so satisfying 👨‍🍳 #cooking #homemade #satisfaction",
    "Gardening is therapeutic. My tomatoes are finally growing 🍅 #gardening #tomatoes #therapy",
    "Photography walk around downtown captured some great shots 📷 #photography #downtown #art",
    "Meditation app reminded me to breathe today 🧘 #meditation #mindfulness #breathe",
    "Knitting scarf for winter. Hands are cramping 🧶 #knitting #winter #handmade",
    # Mixed posts with multiple brands
    "Airport outfit: Nike sneakers, Lululemon leggings, and my trusty Patagonia backpack ✈️ #Nike #Lululemon #Patagonia #travel",
    "Work setup: MacBook Air, AirPods Pro, and way too much Dunkin coffee ☕ #Apple #Dunkin #work #setup",
    "Target run turned into $200 somehow. Also grabbed Chipotle for lunch 🎯 #Target #Chipotle #shopping #oops",
    "Walmart grocery pickup while listening to Spotify podcasts #Walmart #Spotify #groceries #efficient",
    "CVS receipt was 3 feet long for buying Advil and Gatorade #CVS #Advil #Gatorade #receipt #long",
    "Costco bulk shopping with my Honda Pilot packed full #Costco #Honda #bulk #shopping",
    "Whole Foods prices make me appreciate Trader Joe's #WholeFoods #TraderJoes #expensive #groceries",
    "Amazon Prime delivery while wearing my Adidas tracksuit #Amazon #Adidas #delivery #comfort",
    "Home Depot trip in my Ford truck for weekend projects #HomeDepot #Ford #weekend #projects",
    "Best Buy employee recommended Sony headphones over Beats #BestBuy #Sony #Beats #headphones",
    # Posts with misspellings and variations
    "Addidas sneakers are so comfortable! Can't spell but love the brand 😅 #addidas #sneakers",
    "Mc Donalds breakfast sandwich was actually decent today #mcdonalds #breakfast",
    "nike air max are my favorite running shoes #nike #airmax #running",
    "starbucks coffee is overpriced but addictive ☕ #starbucks #coffee #expensive",
    "my iphone screen cracked again 😭 #iphone #cracked #apple",
    "playstation controller died during boss fight #playstation #controller #gaming",
    "netflix password sharing days are over #netflix #password #sharing",
    "amazon delivery was late again #amazon #delivery #late",
    "tesla charging station was broken #tesla #charging #broken",
    "walmart self checkout never works #walmart #selfcheckout #broken",
    # No brands - personal struggles and thoughts
    "Anxiety is rough today. Taking it one breath at a time 💙 #anxiety #mentalhealth #breathe",
    "Student loans are crushing my soul slowly 💸 #studentloans #debt #struggle",
    "Job hunting is soul-crushing. When will it end? 😩 #jobhunt #unemployment #struggle",
    "Adulting is harder than anyone warned me about #adulting #struggle #life",
    "Imposter syndrome hitting hard at work today #impostersyndrome #work #anxiety",
    "Therapy session was heavy but helpful today #therapy #mentalhealth #growth",
    "Depression makes everything feel impossible some days #depression #mentalhealth #struggle",
    "Burnout is real. Need a vacation ASAP #burnout #vacation #tired",
    "Social anxiety makes networking events torture #socialanxiety #networking #introvert",
    "Comparison is the thief of joy. Social media reminder #comparison #socialmedia #mentalhealth",
    # No brands - positive life moments
    "Got promoted at work! Hard work pays off 🎉 #promotion #work #success",
    "Baby said her first word today. My heart melted ❤️ #baby #firstword #parenthood",
    "Anniversary dinner was perfect. Lucky to have found my person 💕 #anniversary #love #grateful",
    "Graduated college today! 4 years flew by 🎓 #graduation #college #achievement",
    "Bought my first house! The American dream is real 🏠 #house #homeowner #dream",
    "Marathon training is paying off. New personal record! 🏃‍♂️ #marathon #training #PR",
    "Volunteer work at the shelter was fulfilling today ❤️ #volunteer #shelter #giving",
    "Surprise birthday party made me cry happy tears 🎂 #birthday #surprise #grateful",
    "Random act of kindness from stranger restored my faith #kindness #stranger #faith",
    "Grandpa's 90th birthday party was beautiful celebration 🎉 #grandpa #birthday #family",
    # More brand posts - retail and shopping
    "Target dollar section got me again. Spent $50 on random stuff 🎯 #Target #impulse #shopping",
    "Ikea furniture assembly took 6 hours and my sanity #Ikea #furniture #assembly #struggle",
    "Costco samples convinced me to buy things I don't need #Costco #samples #impulse",
    "Home Depot employee was super helpful finding screws #HomeDepot #helpful #employee",
    "Lowes price matched Home Depot without hassle #Lowes #HomeDepot #price #match",
    "Walmart self-checkout judged my life choices again #Walmart #selfcheckout #judgment",
    "Best Buy Geek Squad fixed my computer quickly #BestBuy #GeekSquad #computer #repair",
    "GameStop trade-in value was insulting as usual #GameStop #tradein #ripoff",
    "Barnes & Noble cafe has surprisingly good coffee #BarnesNoble #cafe #coffee #books",
    "Whole Foods parking lot is a war zone #WholeFoods #parking #chaos",
    # More no brand posts - random observations
    "People who don't return shopping carts are chaos agents 🛒 #shoppingcarts #chaos #society",
    "Why do all public restrooms have broken soap dispensers? 🧼 #publicrestrooms #soap #broken",
    "Self-checkout machines have trust issues with my bananas 🍌 #selfcheckout #bananas #trust",
    "Elevator small talk should be illegal #elevator #smalltalk #awkward",
    "Morning people are suspicious. What are they hiding? ☀️ #morningpeople #suspicious #night",
    "Parallel parking in the city is extreme sport 🚗 #parallelparking #city #extreme",
    "Grocery store music is designed to make you buy more #grocery #music #psychology",
    "Gas prices make me want to walk everywhere ⛽ #gas #prices #expensive #walking",
    "Traffic lights know when you're running late #traffic #lights #conspiracy #late",
    "WiFi password is always more complicated than necessary 📶 #wifi #password #complicated",
    # More brand posts - food delivery and apps
    "DoorDash driver couldn't find my house again #DoorDash #delivery #lost #driver",
    "Uber Eats delivery fee costs more than the food #UberEats #delivery #expensive #fee",
    "Grubhub recommended the perfect restaurant tonight #Grubhub #recommendation #dinner #perfect",
    "Postmates merged with Uber and I'm still confused #Postmates #Uber #merger #confused",
    "Seamless order arrived cold but I was too hungry to care #Seamless #cold #food #hungry",
    "Food delivery apps are ruining my budget and health #delivery #apps #budget #health",
    "Yelp reviews are either 5 stars or 1 star. No middle ground #Yelp #reviews #extreme",
    "OpenTable reservation saved my anniversary dinner #OpenTable #reservation #anniversary #saved",
    "Skip the Dishes lived up to its name. Order never came #SkipTheDishes #delivery #fail",
    "Favor delivery person was super friendly and fast #Favor #delivery #friendly #fast",
    # No brands - weather and seasons
    "First snow of the season always makes me smile ❄️ #snow #winter #first #smile",
    "Spring allergies are trying to kill me this year 🤧 #spring #allergies #pollen #death",
    "Summer heat wave has me hiding indoors all day 🥵 #summer #heat #hiding #indoors",
    "Fall leaves are peak nature eye candy 🍂 #fall #leaves #nature #beautiful",
    "Rainy Monday mood matches the weather perfectly ☔ #rainy #monday #mood #weather",
    "Humidity makes me question living in this state 💦 #humidity #questioning #state #weather",
    "Perfect beach weather wasted on a work day ☀️ #beach #weather #work #wasted",
    "Storm knocked out power for 6 hours yesterday ⛈️ #storm #power #outage #yesterday",
    "Sunrise this morning was absolutely breathtaking 🌅 #sunrise #morning #breathtaking #beautiful",
    "Fog so thick I can barely see my car 🌫️ #fog #thick #visibility #car",
    # More brand posts - streaming and media
    "Disney Plus has all my childhood movies #DisneyPlus #childhood #movies #nostalgia",
    "HBO Max password sharing with the family #HBO #Max #password #family",
    "Amazon Prime Video recommendations are getting weird #Amazon #Prime #recommendations #weird",
    "Hulu ads interrupt at the worst moments #Hulu #ads #interruption #timing",
    "Apple TV Plus actually has some good shows #Apple #TV #shows #surprising",
    "Peacock streaming service exists apparently #Peacock #streaming #service #forgot",
    "Paramount Plus is just CBS with extra steps #Paramount #Plus #CBS #confusing",
    "Crunchyroll for anime nights with friends #Crunchyroll #anime #friends #nights",
    "ESPN Plus for sports I didn't know existed #ESPN #Plus #sports #obscure",
    "Discovery Plus has weird documentaries at 3am #Discovery #Plus #documentaries #weird",
    # No brands - work and career thoughts
    "Open office layouts were designed by introverts' enemies #office #layout #introverts #enemies",
    "Meetings that could have been emails should be illegal #meetings #emails #illegal #waste",
    "Coworker microwaved fish in the break room again 🐟 #coworker #fish #microwave #smell",
    "Work from home pants are the best invention ever #workfromhome #pants #invention #comfort",
    "Boss replied to email at midnight. Boundaries people! #boss #email #midnight #boundaries",
    "Coffee machine broke and productivity dropped 50% ☕ #coffee #machine #productivity #crisis",
    "Printer jammed 5 minutes before important presentation #printer #jammed #presentation #timing",
    "Elevator pitch turned into elevator awkward silence #elevator #pitch #awkward #silence",
    "Office plant died under my care. Green thumb not found 🪴 #office #plant #death #blackthumb",
    "Lunch break walk cleared my head completely 🚶‍♀️ #lunch #walk #cleared #head",
    # More brand posts - cars and automotive
    "Chevy Silverado handles construction sites perfectly #Chevy #Silverado #construction #perfect",
    "Subaru Outback is the ultimate adventure vehicle #Subaru #Outback #adventure #ultimate",
    "Mazda CX-5 has surprisingly good gas mileage #Mazda #CX5 #gas #mileage",
    "Jeep Wrangler with doors off hits different 🚙 #Jeep #Wrangler #doors #freedom",
    "Volkswagen reliability has improved dramatically #Volkswagen #reliability #improved #dramatically",
    "Hyundai warranty gives me peace of mind #Hyundai #warranty #peace #mind",
    "Kia Soul looks weird but drives great #Kia #Soul #weird #drives",
    "Nissan Altima is the most boring reliable car #Nissan #Altima #boring #reliable",
    "Lexus interior feels like luxury spaceship #Lexus #interior #luxury #spaceship",
    "Acura MDX perfect for family road trips #Acura #MDX #family #roadtrips",
    # No brands - random life observations
    "Socks disappear in the dryer to another dimension 🧦 #socks #dryer #dimension #mystery",
    "Phone battery dies fastest when you need it most 🔋 #phone #battery #timing #murphy",
    "Autocorrect creates more problems than it solves #autocorrect #problems #solutions #technology",
    "Left turn arrows are never long enough #traffic #arrows #timing #frustration",
    "Hotel shower temperature is either arctic or lava 🚿 #hotel #shower #temperature #extreme",
    "Airplane armrest belongs to middle seat person ✈️ #airplane #armrest #middle #seat",
    "Grocery list forgotten at home every single time 📝 #grocery #list #forgotten #home",
    "Keys are always in the last place you look 🔑 #keys #last #place #search",
    "Dentist appointments are scheduled 6 months in advance 🦷 #dentist #appointments #advance #planning",
    "USB cable requires 3 attempts to plug in correctly #USB #cable #attempts #frustration",
    # More brand posts - beauty and personal care
    "Sephora VIB sale emptied my bank account 💄 #Sephora #VIB #sale #bankrupt",
    "Ulta rewards points actually add up to something #Ulta #rewards #points #savings",
    "MAC lipstick survived entire wedding day #MAC #lipstick #wedding #lasting",
    "Fenty Beauty foundation matches my skin perfectly #Fenty #Beauty #foundation #match",
    "Glossier packaging is Instagram-worthy aesthetic #Glossier #packaging #Instagram #aesthetic",
    "Urban Decay eyeshadow palette worth the hype #Urban #Decay #eyeshadow #hype",
    "Clinique moisturizer cleared up my winter skin #Clinique #moisturizer #winter #skin",
    "L'Oreal drugstore dupe works just as well #LOreal #drugstore #dupe #works",
    "Maybelline mascara is ride or die forever #Maybelline #mascara #ride #die",
    "CeraVe cleanser recommended by dermatologists everywhere #CeraVe #cleanser #dermatologist #recommended",
    # No brands - health and fitness
    "Gym motivation disappears after first set 💪 #gym #motivation #disappears #first",
    "Yoga instructor made pretzel shapes look easy 🥨 #yoga #instructor #pretzel #impossible",
    "Running outside beats treadmill every time 🏃‍♂️ #running #outside #treadmill #better",
    "Protein shake tastes like chalk but gains bro #protein #shake #chalk #gains",
    "Rest day guilt is real but necessary #rest #day #guilt #necessary",
    "Stretching feels amazing until you realize flexibility #stretching #amazing #flexibility #lacking",
    "Water bottle reminder app saved my hydration #water #bottle #reminder #hydration",
    "Sleep tracker shows I'm basically nocturnal 😴 #sleep #tracker #nocturnal #schedule",
    "Morning workout energy lasts exactly 2 hours #morning #workout #energy #duration",
    "Foam roller hurts so good after leg day #foam #roller #hurts #good",
    # More brand posts - home and lifestyle
    "Roomba got stuck under couch again #Roomba #stuck #couch #robot",
    "Ring doorbell caught package thief red-handed #Ring #doorbell #package #thief",
    "Nest thermostat learns my schedule better than I do #Nest #thermostat #schedule #smart",
    "Alexa misunderstood my request spectacularly #Alexa #misunderstood #request #spectacular",
    "Google Home plays wrong song every time #Google #Home #wrong #song",
    "Philips Hue lights set perfect movie mood #Philips #Hue #lights #movie",
    "Instant Pot pressure cooker changed my cooking game #Instant #Pot #pressure #cooking",
    "KitchenAid mixer is the kitchen workhorse #KitchenAid #mixer #kitchen #workhorse",
    "Dyson vacuum picks up cat hair like magic #Dyson #vacuum #cat #hair",
    "Shark steam mop makes cleaning almost enjoyable #Shark #steam #mop #cleaning",
    # No brands - food and cooking experiences
    "Burned dinner while scrolling social media again 🔥 #burned #dinner #social #media",
    "Recipe said 20 minutes but took 2 hours somehow ⏰ #recipe #time #cooking #reality",
    "Onions make me cry but taste so good 🧅 #onions #cry #taste #good",
    "Leftover pizza for breakfast is perfectly acceptable 🍕 #leftover #pizza #breakfast #acceptable",
    "Cooking from scratch feels like major accomplishment 👨‍🍳 #cooking #scratch #accomplishment #pride",
    "Kitchen timer went off and I have no memory setting it ⏰ #kitchen #timer #memory #confusion",
    "Spice level 'mild' at restaurant destroyed my mouth 🌶️ #spice #mild #restaurant #destroyed",
    "Meal prep Sunday lasted exactly one week #meal #prep #sunday #lasted",
    "Garlic breath is small price for delicious food 🧄 #garlic #breath #price #delicious",
    "Food delivery versus cooking at home eternal struggle #food #delivery #cooking #struggle",
    # Final batch - mixed content to reach target
    "Package arrived broken thanks to rough handling 📦 #package #broken #rough #handling",
    "Neighbor's dog barks at 5am every morning 🐕 #neighbor #dog #barks #morning",
    "Library late fees cost more than buying the book 📚 #library #late #fees #expensive",
    "Ice cream truck song triggers childhood memories 🍦 #ice #cream #truck #childhood",
    "Construction noise starts exactly when I fall asleep 🚧 #construction #noise #sleep #timing",
    "Farmer's market vegetables taste like actual food 🥬 #farmers #market #vegetables #real",
    "Bus schedule is more like a rough suggestion 🚌 #bus #schedule #suggestion #unreliable",
    "Sidewalk chalk art washes away too quickly 🎨 #sidewalk #chalk #art #temporary",
    "Crosswalk button might be placebo effect #crosswalk #button #placebo #effect",
    "Speed bump designed by someone who hates cars #speed #bump #designed #hate",
    # Even more posts to reach the target count
    "Morning coffee ritual is sacred and non-negotiable ☕ #morning #coffee #ritual #sacred",
    "Cat knocked over plant again. Chaos agent confirmed 🐱 #cat #plant #chaos #agent",
    "Weekend plans: absolutely nothing and loving it #weekend #plans #nothing #loving",
    "Laundry pile reached critical mass yesterday 👕 #laundry #pile #critical #mass",
    "Dishwasher broke and suddenly appreciate modern convenience #dishwasher #broke #modern #convenience",
    "Mailbox key mystery: how do people lose these? 📮 #mailbox #key #mystery #lost",
    "Parallel universe where matching socks exist #parallel #universe #matching #socks",
    "Remote control disappeared into couch dimension again 📺 #remote #control #couch #dimension",
    "Plant care instructions: water weekly. Plant: dies anyway 🪴 #plant #care #water #death",
    "Doorbell rings during nap time without fail 🛎️ #doorbell #nap #time #timing",
    # More brand posts to balance the dataset
    "Victoria's Secret Pink collection never goes on sale #Victoria #Secret #Pink #sale",
    "Bath & Body Works candles smell like happiness #Bath #Body #Works #candles",
    "Old Navy clearance rack is treasure hunting #Old #Navy #clearance #treasure",
    "Gap jeans fit perfectly after years of searching #Gap #jeans #fit #perfect",
    "American Eagle jeans stretch in all right places #American #Eagle #jeans #stretch",
    "Hollister cologne smell transports to high school #Hollister #cologne #high #school",
    "Abercrombie rebrand actually worked surprisingly well #Abercrombie #rebrand #worked #surprising",
    "Forever 21 sizing makes no sense whatsoever #Forever #21 #sizing #nonsense",
    "Express work clothes survive office life #Express #work #clothes #survive",
    "Banana Republic quality justifies higher prices #Banana #Republic #quality #prices",
    # Technology and gadgets with brands
    "Canon camera captures memories perfectly every time #Canon #camera #memories #perfect",
    "Nikon lens quality worth the investment price #Nikon #lens #quality #investment",
    "GoPro survived underwater adventure unscathed #GoPro #underwater #adventure #survived",
    "Fitbit tracks steps but judges life choices #Fitbit #steps #judges #choices",
    "Garmin GPS never led me astray yet #Garmin #GPS #never #astray",
    "Bose headphones cancel noise and stress #Bose #headphones #cancel #stress",
    "JBL speaker brings party wherever I go #JBL #speaker #party #portable",
    "Roku streaming device simplifies entertainment choices #Roku #streaming #simplifies #entertainment",
    "Chromecast makes any TV smart instantly #Chromecast #TV #smart #instant",
    "Fire Stick interface could use improvement #Fire #Stick #interface #improvement",
]

system_prompt = """
# Role and Objective
You are a brand extraction specialist. Your task is to analyze social media posts and extract all brand names mentioned, then return them in CSV format.

## Instructions:
1. **Analyze the given social media post** for any brand names, including:
- Product brands (Nike, Apple, Coca-Cola)
- Company names (Google, Microsoft, Tesla)
- Service brands (Uber, Netflix, Spotify)
- Fashion/luxury brands (Gucci, Louis Vuitton, Zara)
- Food/restaurant brands (McDonald's, Starbucks, KFC)
- Tech brands and apps (Instagram, TikTok, iPhone)
- Retail brands (Amazon, Target, Walmart)
- Other things that are brands

2. **Extraction Rules:**
- Extract exact brand names as they appear in the post
- Include brands mentioned directly or through hashtags (#Nike, #Starbucks)
- Include brands mentioned through product names (iPhone = Apple brand)
- Do NOT extract generic terms, common words, or personal names
- Do NOT extract location names unless they are clearly brand names
- Include misspelled brand names if the intent is clear

3. **Output Format:**
Return results as a comma-separated list of brand names:
- Single brand: `Nike`
- Multiple brands: `Nike,Starbucks,Apple`
- No brands found: None

## Examples:
**Input:** "Just got my new iPhone 15 from Apple Store! #apple #tech Love how it pairs with my AirPods"
**Output:** Apple

**Input:** "Coffee run at Starbucks before hitting the Nike store. #starbucks #nike #shopping"
**Output:** Starbucks,Nike

**Input:** "Beautiful sunset today, feeling grateful"
**Output:** None

Now analyze the social media post and extract brands in CSV format."""


# =============================================================================
# LOCUST USER CLASS - This defines what each simulated user does
# =============================================================================


class LLMUser(HttpUser):
    """
    Each instance represents one simulated user making requests to your LLM endpoints.
    Locust will create multiple instances of this class to simulate concurrent load.
    """

    wait_time = between(0, 0.3)  # Wait 0-0.3 seconds between requests per user
    host = config["host_url"]

    def on_start(self):
        """
        Called once when each user starts up.
        Creates an OpenAI client that will be reused for all requests by this user.
        """
        self.client = OpenAI(
            base_url=config["host_url"],
            api_key=config["api_key"],
            timeout=60.0 * 60.0,  # 1 hour timeout for long requests
        )

    def execute_task(self, task_name):
        """
        Execute one LLM inference request and record performance metrics.
        This is where the actual work happens - calling your deployed model.
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=config["model"],
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": random.choice(POSTS)},
                ],
            )

            total_time = int(
                (time.time() - start_time) * 1000
            )  # Convert to milliseconds

            # Log successful request to Locust dashboard
            self.environment.events.request.fire(
                request_type="OpenAICompatibleServer",
                name=f"task_{task_name}",
                response_time=total_time,  # Latency in ms
                response_length=len(response.choices[0].message.content),
                exception=None,
                context={"task": task_name},
            )

        except Exception as e:
            # Log failed request to Locust dashboard
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="OpenAICompatibleServer",
                name=f"task_{task_name}",
                response_time=total_time,
                response_length=0,
                exception=e,
                context={"task": task_name},
            )

    @task
    def llm_task(self):
        """
        Main task method - called repeatedly by Locust for each user.
        Each call represents one request cycle:
        1. Choose a task to run
        2. Execute it
        3. Wait (based on wait_time)
        4. Repeat
        """
        self.execute_task(task_name)
```

Run the benchmark with:

`uv run locust -f locustfile.py --web-port 8089`

Then open http://localhost:8089 to start the test via web UI.

