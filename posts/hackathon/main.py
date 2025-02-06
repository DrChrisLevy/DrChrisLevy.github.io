import json
import os

import modal
from dotenv import load_dotenv
from modal import Image, enter

load_dotenv()


# class Post(BaseModel):
#     caption: str
#     thumbnail: str
#     media_url: str
#     type: Literal["IMAGE", "VIDEO"]
#     date: datetime
#     post_id: str


# class CreatorData(BaseModel):
#     creator_handle: str
#     posts: List[Post]
#     brand_competitors: List[str]


app = modal.App("hackathon")

image = Image.debian_slim(python_version="3.11").pip_install("openai", "python-dotenv")

vol = modal.Volume.from_name("hackathon-vol", create_if_missing=True)


@app.cls(image=image, volumes={"/data": vol}, cpu=4, timeout=600, container_idle_timeout=300, secrets=[modal.Secret.from_dotenv()])
class Model:
    @enter()
    def setup(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.completion = self.client.chat.completions.create
        self.cache = True

    @modal.web_endpoint(method="POST", docs=True)
    def creator_mention_buckets(self, data: dict):
        # I want to read the data from the volume if it exists
        if self.cache and os.path.exists(f"/data/{data['creator_handle']}.json"):
            with open(f"/data/{data['creator_handle']}.json", "r") as f:
                return json.load(f)

        over_all_summary = "overall summary"
        brands = self.find_brands([p["caption"] for p in data["posts"]])
        buckets = self.bucket_posts([p for p in data["posts"]], brands)
        final_buckets = self.label_buckets(data["creator_handle"], buckets)

        resp = {
            "creator_handle": data["creator_handle"],
            "buckets": final_buckets,
            "summary": over_all_summary,
        }
        # I want to write the response to a JSON file in the volume
        # The name of the file should be the creator_handle
        with open(f"/data/{data['creator_handle']}.json", "w") as f:
            json.dump(resp, f)
        return resp

    def find_brands(self, captions: list[str]):
        all_captions = "\n".join(captions)
        response = self.completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """"
                 Role:
                 
                 You are an expert at extracting brands from a list of social media brand captions.
                 The post captions come from an influencer/creator. 
                 
                 Instructions:
                 Your role is to extract the brands the influencer/creator is talking about in their captions.
                 Provide a comma separated list of brands mentioned in the posts by the influencer/creator.

                **IMPORTANT**: Only extract legit brands/account/competitors. Don't extract common keywords or product names.
                 
                 Expected Output: (dont put "and" or "or" in the output)
                 brand1, brand2, brand3, brand4, brand5
                 
                 """,
                },
                {
                    "role": "user",
                    "content": f"Here are the posts. Extract the brands and return only a comma separated list of brands, nothing else:\n\n {all_captions}",
                },
            ],
        )
        return [b.lower().strip() for b in response.choices[0].message.content.split(",")]

    def bucket_posts(self, posts: list[dict], brands: list[str]):
        buckets = {}
        for post in posts:
            for brand in brands:
                if f" {brand.lower()} " in post["caption"].lower():
                    buckets[brand] = buckets.get(brand, []) + [post]
        buckets = {k: v for k, v in buckets.items() if len(v) >= 2}
        return buckets

    def label_bucket(self, creator_handle: str, brand_name: str, captions: list[str]):
        captions = "\n".join([p for p in captions])
        response = self.completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """"
                 Role:
                 
                 You are an expert at extracting a short vibe summary from a list of captions where
                 an influencer/creator is talking about a brand. You will also provide a vibe classification/score for the brand.
                 
                 Instructions:
                 Write a short summary of the influencer/creator's vibe towards the brand following the format below.
                 Also classify the vibe into one of the following four categories: High, Mid, Low, Trash
                 
                Format Examples
                Here are some examples where a creator would be talking about the brand nike. These are examples
                of how you should write the vibe summary for the data you are given. Only provide one summary.

                summary: When mentioning @nike, the creator loves the style of the footwear, specifically the Nike Air range.
                vibes: High

                summary: When mentioning @nike, the creator talks favorably about their color options and size range, but would like better arch support.
                vibes: Mid

                summary: When mentioning @nike, the creator mentioned how expensive the shoes were, and that they were a free gift but they would not purchase on their own.
                vibes: Low

                summary: When mentioning @nike, the creator talks in length about how difficult it was to get customer service, and how many blisters they got from the shoes.
                vibes: Trash

                Final Output:
                You always return JSON output with the two keys: summary and vibes.
                {
                    "summary": "<summary>",
                    "vibes": "<vibes>"
                }
                 
                 """,
                },
                {
                    "role": "user",
                    "content": f"Here are the posts where the creator/influencer {creator_handle} is talking about the brand {brand_name}. Provide the vibe summary and vibe classification:\n\n {captions}",
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def label_buckets(self, creator_handle: str, buckets: dict[str, list[dict]]):
        import time

        ct = time.time()
        from concurrent import futures

        def process_bucket(brand_data):
            brand_name, posts = brand_data
            captions = [p["caption"] for p in posts]
            post_ids = [p["post_id"] for p in posts]
            lab = self.label_bucket(creator_handle, brand_name, captions)
            return {
                "brand": brand_name,
                "post_ids": post_ids,
                "summary": lab["summary"],
                "vibes": lab["vibes"],
            }

        with futures.ThreadPoolExecutor(max_workers=30) as executor:
            final_buckets = list(executor.map(process_bucket, buckets.items()))
        print(f"Time taken: {time.time() - ct}")
        return final_buckets
