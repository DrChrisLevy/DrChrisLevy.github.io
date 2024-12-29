import json
import os
import random
import time
from uuid import uuid4

import anthropic
import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import upload_file
from openai import OpenAI
from sqlitedict import SqliteDict

system_prompt = """

You are a social media content generation expert tasked with creating realistic social media posts for synthetic user profiles. 
Each user is focused on a specific topic or area of interest (label). 
Your task is to produce a large number of tweets for a single user under a single label, demonstrating a consistent style and persona.

Here are all the available labels for context:

**Tech & Business**

- Tech Industry Analysis
- Software Engineering
- Frontend Development
- Data Analytics
- AI & Machine Learning
- Cybersecurity News
- Cryptocurrency & Web3
- Web3 Innovation
- NFT Trading
- Startup Ecosystem
- Venture Capital Analysis
- Paid Advertising
- Content Marketing
- E-commerce Innovation
- Business Leadership
- Product Management
- Fintech Discussion
- Sales Strategy
- Tech Entrepreneurship

**Politics & News**

- US Politics Analysis
- Global Affairs Commentary
- Electoral Politics
- Political Commentary
- Legal System Analysis
- Military & Defense
- Climate Change Discussion
- Economic Policy
- Political Satire
- Local Community News

**Entertainment & Media**

- Film & Cinema Analysis
- TV Series Discussion
- Reality TV Commentary
- Music Industry Analysis
- Video Content Creation
- Video Game Enthusiast
- Competitive Gaming
- Indie Game Dev
- Anime & Manga Community
- Comics & Graphic Novels
- Celebrity Commentary
- Fashion & Streetwear
- Sneaker Culture
- Book & Literature
- Podcast Creation
- Entertainment Industry
- Live Music Fan

**Sports**

- NFL Analysis
- NBA Discussion
- MLB Commentary
- Soccer Coverage
- Formula 1 Community
- College Sports Analysis
- MMA & Boxing
- Weightlifting Training
- Fitness Training
- Endurance Sports
- Sports Betting
- Olympics Coverage

**Science & Education**

- Space Exploration
- Biology Research
- Physics Discussion
- Health & Medicine
- EdTech Innovation
- Historical Analysis
- Psychology Research
- Environmental Science
- Earth Sciences
- Academic Research

**Lifestyle & Interests**

- Travel Photography
- Food & Cooking
- Professional Photography
- Amateur Photography
- Home Improvement
- Home Gardening
- Investment Strategy
- Personal Investing
- Pet Community
- Meditation Practice
- Digital Art
- Visual Arts
- Automotive Culture
- Craft Beer Culture
- Coffee Enthusiasm
- Culinary Arts

**Communities & Culture**

- Parenting Discussion
- Mental Health Support
- Spiritual Practice
- Philosophy Discussion
- Urban Culture
- Vintage Collection
- DIY Crafts
- Language Learning
- Open Source Coding
- Personal Development
- Minimalist Living
- Sustainable Living
- Fiction Writing
- Conspiracy Theories
- Fan Culture
- Internet Culture
- Outdoor Adventure
- Alternative Lifestyle

**Platform & Meta**

- Twitter Meta Commentary
- Meme Creation
- Viral Content
- Personal Updates
- Social Commentary
- Community Building
- Twitter Spaces Hosting
- Platform Critique
- Bot & Automation
- Online Privacy
- Data Visualization

Instructions:

* Do not include conversation. Focus on the user's opinion, commentary, or updates.
* Each tweet should feel distinct and not merely repeat the same ideas or concepts.
* Incorporate common social media elements like hashtags, @mentions (of real life users or fictional), links to relevant articles and use of emojis.
* Each user should have a single primary area of focus(label), but can include other relevant content related to that area.
* The tone should feel natural and conversational.
* Use realistic slang and phrases, when appropriate, to give the user a real voice.
* When referencing news articles or events, use a mix of real events in your training data as well as fake events you make up.
* Generate a substantial amount of content - approximately (aim for ~50-75 unique tweets) per user in order to achieve good diversity.
* Assume the tweets are from a single user over a few months, so there may be a variance in what they post and how they post.
* Do not introduce conversational elements in the posts (such as questions). The posts are meant to be a user's view of the world.

Now, create the profile of a single social media user with the following attributes.

Use the following parameters to guide the creation of the user:
* **Assigned Label:** [INSERT_LABEL_HERE]
* **Demographics:** Create plausible demographics (age, gender, location, profession) that align with the assigned label. Be specific.
* **Personality:** Create a distinct personality type for this user (e.g., analytical, humorous, sarcastic, enthusiastic, etc.). Be specific.
* **Viewpoint/Bias:** Create a general viewpoint or bias this user might have as it relates to their selected label.
* **Posting Style:** Create a particular way this user might post. For example, do they use lots of emojis? Do they tend to be more formal or informal? Do they prefer short or long posts?

Do not use the user's persona in the content of the tweets. The tweets should come from the user, not talk about the user. But the persona should come through in the tweets. For example, some of the tweets can mention the users location etc.

Based on the user profile you just created and the system prompt, generate a large number of individual social media posts for this user.

Focus primarily on posts that align with the assigned label.

Generate approximately 50-75 unique posts per user.

Do not add numbers to the posts.

You are to output JSON output with two fields. One for the "persona" and one for the "tweets".

Here are some smaller examples to show the format. The examples you generated will have many more tweets though. These are purely to show the JSON format.

Examples:

{
  "persona": {
    "label": "Coffee Enthusiasm",
    "demographics": {
      "age": 32,
      "gender": "Female",
      "location": "Portland, OR",
      "profession": "Independent Coffee Roaster"
    },
    "personality": "Passionate and detail-oriented with a touch of dry humor. Takes coffee seriously but not herself.",
    "viewpoint": "Believes in sustainable sourcing and artisanal production methods. Skeptical of large chain coffee shops.",
    "postingStyle": "Uses rich descriptive language, occasional industry jargon, and coffee-themed emojis. Loves sharing brewing tips and origin stories."
  },
  "tweets": [
    "Just cupped our new Ethiopian Yirgacheffe lot. The jasmine notes are absolutely singing this morning ☕ These beans are telling a story and I'm here for every chapter",
    "That moment when someone calls your light roast 'under-developed' and you have to explain that not everything needs to taste like charcoal 🔥 #SpecialtyCoffee #RoasterLife",
    "Big thanks to @FarmersCollective for another amazing micro-lot! Article on their innovative processing methods: coffeeweekly.com/innovations-in-natural-processing [fictional link] 🌱",
    "Pro tip: If your coffee tastes bitter, check your water temperature. 205°F is not always your friend, folks! Sometimes 195°F lets those subtle notes shine through ✨ #BrewingTips",
    "Just installed our new @Loring roaster and wow - the heat control is next level. Time to dial in those profiles! 🎯 #CoffeeRoasting",
    "Random coffee fact: The average coffee tree produces only 1-1.5 pounds of roasted coffee annually. Think about that next time you're complaining about specialty prices 😅"
  ]
}


{
  "persona": {
    "label": "Formula 1 Community",
    "demographics": {
      "age": 28,
      "gender": "Male",
      "location": "Montreal, Canada",
      "profession": "Motorsport Engineer"
    },
    "personality": "Technical and analytical, with occasional bursts of excitement during race weekends. Lives for the data and technical regulations.",
    "viewpoint": "Strong believer in engineering innovation. Critical of restrictive regulations but supportive of safety measures.",
    "postingStyle": "Uses lots of technical jargon, racing terminology, and track emojis. Often shares detailed analysis with graphs and technical breakdowns."
  },
  "tweets": [
    "These new floor edge regulations are fascinating. Looking at the Mercedes solution vs Ferrari's approach, it's clear there's still room for interpretation 🏎️ #F1Tech",
    "Brilliant qualifying from @MaxVerstappen33! That last sector management was *chef's kiss* Perfect example of managing tire temps through the lap 🌡️",
    "Running the numbers on tire degradation from FP2. Ferrari looking strong on mediums but that high wear rate could hurt them in the final stint 📊 ",
    "The way Alpine has packaged their sidepods this year is absolutely brilliant. Old school cooling efficiency with modern aero principles. Engineering at its finest! 🛠️",
    "Hot take: DRS zones shouldn't be standardized. Each track needs custom activation points based on corner exit speed and straight length. Would make for much better racing 🏁"
  ]
}
  
{
  "persona": {
    "label": "Space Exploration",
    "demographics": {
      "age": 35,
      "gender": "Female",
      "location": "Houston, Texas",
      "profession": "Aerospace Systems Engineer"
    },
    "personality": "Enthusiastic science communicator with a deep love for space history. Optimistic about humanity's future in space.",
    "viewpoint": "Believes in public-private partnership for space exploration. Advocates for increased science funding.",
    "postingStyle": "Combines technical accuracy with accessible explanations. Uses space emojis and shares lots of visual content links."
  },
  "tweets": [
    "The James Webb's latest deep field image is just... I've been staring at it for an hour. Each point of light is an entire galaxy. We are so small and yet so capable of understanding the vast 🌌",
    "BREAKING: New paper just dropped on potential biosignatures in Europa's spectral data! This is preliminary but incredibly exciting spacejournal.com/europa-analysis 🛸",
    "Fun fact: The Saturn V rocket's F-1 engines were so powerful that each one could drain an Olympic swimming pool in just 30 seconds! 🚀 #SpaceHistory",
    "Watching today's spacewalk and still amazed by the engineering that goes into EMU design. These suits are basically tiny spaceships! Live feed analysis: spacefeed.com/eva-294",
    "Remember when we thought Mars was covered in canals? Our understanding of the solar system has come so far, and we're just getting started 🌎"
  ]
}

{
  "persona": {
    "label": "Indie Game Dev",
    "demographics": {
      "age": 25,
      "gender": "Non-binary",
      "location": "Berlin, Germany",
      "profession": "Independent Game Developer"
    },
    "personality": "Creative and community-oriented. Openly shares development process and struggles. Supportive of other indie devs.",
    "viewpoint": "Advocates for sustainable development practices and innovative game design. Critical of predatory monetization.",
    "postingStyle": "Casual and authentic. Uses pixel art emojis, shares development screenshots, and frequently engages with the indie community."
  },
  "tweets": [
    "Finally fixed that particle system bug that's been haunting me for weeks! Turns out you should actually read the documentation sometimes 😅 #gamedev #indiedev",
    "Current state of the project: 3,427 cups of coffee, 52 scrapped features, and one very stubborn lighting system that refuses to work as intended 🎮",
    "Just published my shader tutorial! Hope it helps other solo devs: gamedevguide.com/custom-shadersShow me what you create with it! 🎨",
    "Hot take: Your first game doesn't need to be perfect. It needs to be finished. Ship it, learn from it, make the next one better 🚀 #indiedevtips",
    "That feeling when your procedural generation creates something unexpectedly beautiful... This is why I love gamedev ✨"
  ]
}

"""

labels = [
    "Tech Industry Analysis",
    "Software Engineering",
    "Frontend Development",
    "Data Analytics",
    "AI & Machine Learning",
    "Cybersecurity News",
    "Cryptocurrency & Web3",
    "Web3 Innovation",
    "NFT Trading",
    "Startup Ecosystem",
    "Venture Capital Analysis",
    "Paid Advertising",
    "Content Marketing",
    "Ecommerce Innovation",
    "Business Leadership",
    "Product Management",
    "Fintech Discussion",
    "Sales Strategy",
    "Tech Entrepreneurship",
    "US Politics Analysis",
    "Global Affairs Commentary",
    "Electoral Politics",
    "Political Commentary",
    "Legal System Analysis",
    "Military & Defense",
    "Climate Change Discussion",
    "Economic Policy",
    "Political Satire",
    "Local Community News",
    "Film & Cinema Analysis",
    "TV Series Discussion",
    "Reality TV Commentary",
    "Music Industry Analysis",
    "Video Content Creation",
    "Video Game Enthusiast",
    "Competitive Gaming",
    "Indie Game Dev",
    "Anime & Manga Community",
    "Comics & Graphic Novels",
    "Celebrity Commentary",
    "Fashion & Streetwear",
    "Sneaker Culture",
    "Book & Literature",
    "Podcast Creation",
    "Entertainment Industry",
    "Live Music Fan",
    "NFL Analysis",
    "NBA Discussion",
    "MLB Commentary",
    "Soccer Coverage",
    "Formula 1 Community",
    "College Sports Analysis",
    "MMA & Boxing",
    "Weightlifting Training",
    "Fitness Training",
    "Endurance Sports",
    "Sports Betting",
    "Olympics Coverage",
    "Space Exploration",
    "Biology Research",
    "Physics Discussion",
    "Health & Medicine",
    "EdTech Innovation",
    "Historical Analysis",
    "Psychology Research",
    "Environmental Science",
    "Earth Sciences",
    "Academic Research",
    "Travel Photography",
    "Food & Cooking",
    "Professional Photography",
    "Amateur Photography",
    "Home Improvement",
    "Home Gardening",
    "Investment Strategy",
    "Personal Investing",
    "Pet Community",
    "Meditation Practice",
    "Digital Art",
    "Visual Arts",
    "Automotive Culture",
    "Craft Beer Culture",
    "Coffee Enthusiasm",
    "Culinary Arts",
    "Parenting Discussion",
    "Mental Health Support",
    "Spiritual Practice",
    "Philosophy Discussion",
    "Urban Culture",
    "Vintage Collection",
    "DIY Crafts",
    "Language Learning",
    "Open Source Coding",
    "Personal Development",
    "Minimalist Living",
    "Sustainable Living",
    "Fiction Writing",
    "Conspiracy Theories",
    "Fan Culture",
    "Internet Culture",
    "Outdoor Adventure",
    "Alternative Lifestyle",
    "Twitter Meta Commentary",
    "Meme Creation",
    "Viral Content",
    "Personal Updates",
    "Social Commentary",
    "Community Building",
    "Twitter Spaces Hosting",
    "Platform Critique",
    "Bot & Automation",
    "Online Privacy",
    "Data Visualization",
]

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in id2label.items()}

load_dotenv()

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
deepseek_client = OpenAI(
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def query_openai():
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": f'Generate the persona and tweets for the label: "{random.choice(labels)}"'},
        ],
        temperature=random.uniform(0.2, 0.8),
        response_format={"type": "json_object"},
        max_completion_tokens=8000,
    )

    return completion.choices[0].message.content


def query_anthropic():
    label = random.choice(labels)
    prefill = f'{{\n"persona": {{\n    "label": "{label}",\n    "demographics": {{'
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8000,
        temperature=random.uniform(0.5, 1.0),
        system=system_prompt,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": f'Generate the persona and tweets for the label: "{label}"'}]},
            {"role": "assistant", "content": prefill},
        ],
    )
    return prefill + message.content[0].text


def query_google():
    response = google_client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=f'Generate the persona and tweets for the label: "{random.choice(labels)}"',
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=random.uniform(0.8, 1.3),
            max_output_tokens=8000,
        ),
    )
    return response.text


def query_deepseek():
    completion = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Generate the persona and tweets for the label: "{random.choice(labels)}"'},
        ],
        temperature=random.uniform(0.2, 1),
        response_format={"type": "json_object"},
        max_completion_tokens=8000,
    )

    return completion.choices[0].message.content


def generate_data():
    db = SqliteDict("data.db", autocommit=True)
    counter = 0
    while True:
        try:
            db[str(uuid4())] = {"model": "gpt-4o-mini", "data": query_openai()}
            db[str(uuid4())] = {"model": "claude-3-5-sonnet-20241022", "data": query_anthropic()}
            db[str(uuid4())] = {"model": "gemini-2.0-flash-exp", "data": query_google()}
            db[str(uuid4())] = {"model": "deepseek-chat-v3", "data": query_deepseek()}
        except Exception as e:
            print(e)
            time.sleep(5)

        counter += 1
        if counter % 10 == 0:
            print(f"{counter=}")
            print(f"Total entries: {len(db)}")


def push_ds_to_hf():
    db = SqliteDict("data.db")
    records = []
    for r in [v for v in db.values()]:
        try:
            data = json.loads(r["data"])
            r.pop("data")
            r.update({k: v for k, v in data["persona"]["demographics"].items()})
            data["persona"].pop("demographics")
            r.update({k: v for k, v in data["persona"].items()})
            r.update({"tweets": data["tweets"]})
            r["tweets"] = "\n\n".join(r["tweets"])
            r["text"] = r["tweets"]
            r["target_name"] = r["label"]
            r["label"] = label2id[r["label"]]
            r.pop("tweets")
            records.append(r)
        except Exception:
            continue

    # train validation test split
    random.shuffle(records)
    train_records = records[: int(len(records) * 0.8)]
    val_records = records[int(len(records) * 0.8) : int(len(records) * 0.9)]
    test_records = records[int(len(records) * 0.9) :]

    # Create separate datasets for each split
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_records))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_records))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_records))

    # Create DatasetDict and push to hub
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    readme_content = f"""---
license: apache-2.0
task_categories:
- text-classification
language:
- en
tags:
- synthetic
---

    # Synthetic Social Persona Tweets Dataset

    This dataset contains synthetic social media posts generated by various language models.
    This dataset is only meant to be used for quick and dirty experiments i.e. it's a toy dataset.
    Every column/field in this dataset is generated by an LLM.
    The code/prompts used to create this dataset can be found [here](https://github.com/DrChrisLevy/DrChrisLevy.github.io/blob/main/posts/modern_bert/prompts.py).
    The dataset was built to be used for some fine-tuning experiments with ModernBert for one of my blog posts/tutorials.

    Each row in the dataset is a different social media user/persona.
    The `text` field is a concatenated list of posts (textual content) and the `target_name` field is the label of the user/persona.
    The `label` field is the integer id of the `target_name`/label.

    The `id2label` is defined as follows:

    ```python
    {json.dumps(id2label, indent=4)}
    ```
    """
    dataset_dict.push_to_hub("chrislevy/synthetic_social_persona_tweets")

    # Then create the readme
    with open("README.md", "w") as f:
        f.write(readme_content)

    # Upload the readme using huggingface_hub
    upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id="chrislevy/synthetic_social_persona_tweets", repo_type="dataset")
