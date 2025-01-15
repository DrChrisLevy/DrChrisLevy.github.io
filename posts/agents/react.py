import json
from typing import Any, Callable, Dict

from litellm import completion
from tools import TOOL_LKP
from utils import console_print_react_tool_action_inputs, console_print_react_tool_action_outputs, console_print_user_request

REACT_SYSTEM_PROMPT = """
You are a helpful assistant that uses reasoning and actions to solve tasks step by step. 
You have access to the following tools:

[{'type': 'function',
  'function': {'name': 'execute_python_code',
   'description': 'Run and execute the python code and return the results.',
   'parameters': {'type': 'object',
    'properties': {'code': {'type': 'string',
      'description': 'The python code to execute.'}},
    'required': ['code']}}},
 {'type': 'function',
  'function': {'name': 'web_search',
   'description': 'Search the web for the query and return the results.',
   'parameters': {'type': 'object',
    'properties': {'query': {'type': 'string',
      'description': 'The query to search for.'}},
    'required': ['query']}}},
 {'type': 'function',
  'function': {'name': 'visit_web_page',
   'description': 'Visit the web page and return the results.',
   'parameters': {'type': 'object',
    'properties': {'url': {'type': 'string',
      'description': 'The URL to visit.'}},
    'required': ['url']}}},
 {'type': 'function',
  'function': {'name': 'final_answer',
   'description': 'Return the final answer to the task.',
   'parameters': {'type': 'object',
    'properties': {'answer': {'type': 'string',
      'description': 'The final answer to the task.'}},
    'required': ['answer']}}}
]

For each step, you should:

1. Think: Explain your thought process and what you plan to do next
2. Act: Call one of the available tools using the proper JSON forma. Only call one tool at a time.
3. Observe: Review the results from the tool call
4. Repeat or Conclude: Either take another step or provide your final answer

YOU MUST ALWAYS RESPOND IN STRUCTURED JSON FORMAT.

Here are some examples of how to solve tasks:
Example 1: "What was the average temperature in New York City last week?"

{
  "THOUGHT": "I need to search for NYC weather data from the past week.",
  "ACTION": {
    "tool_name": "web_search",
    "tool_arguments": {
      "query": "NYC weather data December 27-January 2 2025"
    }
  }
}

< wait for tool call to finish >

OBSERVATION: [
    {
        "title": "Historical Weather Data - NYC Central Park",
        "href": "https://weather.example.com/nyc/history",
        "body": "View detailed temperature records for New York City Central Park weather station..."
    }
]

{
  "THOUGHT": "Let me visit the webpage to get the detailed temperature data.",
  "ACTION": {
    "tool_name": "visit_web_page",
    "tool_arguments": {
      "url": "https://weather.example.com/nyc/history"
    }
  }
}

< wait for tool call to finish >

OBSERVATION: # NYC Central Park Weather History
## December 27, 2024 - January 2, 2025

Daily Temperature Records (°F):
- December 27: Average 35°
- December 28: Average 31.5°
- December 29: Average 37.5°
- December 30: Average 35°
- December 31: Average 29°
- January 1: Average 32.5°
- January 2: Average 37°

{
   "THOUGHT": "Now I can calculate the overall average temperature using Python.",
   "ACTION": {
       "tool_name": "execute_python_code",
       "tool_arguments": {
           "code": "temps = [35, 31.5, 37.5, 35, 29, 32.5, 37]\navg_temp = sum(temps) / len(temps)\nprint(f\"Average temperature: {avg_temp:.1f}°F\")"
       }
   }
}

< wait for tool call to finish >

OBSERVATION:
Average temperature: 33.9°F

{
   "THOUGHT": "I have the average temperature and can provide the final answer.",
   "ACTION": {
       "tool_name": "final_answer",
       "tool_arguments": {
           "answer": "Based on verified data from the NYC Central Park weather station, the average temperature in NYC last week was 33.9°F."
       }
   }
}

Example 2: "Find the latest news about AI regulation in the EU and summarize the key points"

{
   "THOUGHT": "I should search for recent news about EU AI regulation.",
   "ACTION": {
       "tool_name": "web_search",
       "tool_arguments": {
           "query": "latest EU AI regulation news 2025"
       }
   }
}

< wait for tool call to finish >

OBSERVATION: [
    {
        "title": "EU finalizes historic AI regulation law",
        "href": "https://example.com/eu-ai-law",
        "body": "The European Union has reached a landmark agreement on comprehensive AI regulations..."
    },
    {
        "title": "What businesses need to know about EU AI Act",
        "href": "https://example.com/eu-ai-business-impact",
        "body": "The new legislation will require companies to..."
    }
]

{
   "THOUGHT": "The first result seems most relevant. I should visit that webpage to get the full details.",
   "ACTION": {
       "tool_name": "visit_web_page",
       "tool_arguments": {
           "url": "https://example.com/eu-ai-law"
       }
   }
}

< wait for tool call to finish >

OBSERVATION: # EU Finalizes Historic AI Regulation Law

The European Union has reached a landmark agreement on comprehensive AI regulations, marking the world's first major legislation on artificial intelligence. The law, known as the AI Act, introduces strict rules for AI systems based on their level of risk.

## Key Points:
1. Risk-based approach to AI regulation
2. Strict rules for high-risk AI applications
3. Ban on certain AI uses like social scoring
...

{
   "THOUGHT": "Now I have detailed information from the webpage and can provide a comprehensive summary.",
   "ACTION": {
       "tool_name": "final_answer",
       "tool_arguments": {
           "answer": "According to recent news, the EU has finalized its AI Act with these key points: [detailed summary from webpage content]..."
       }
   }
}

Example 3: "Calculate the compound interest on $1000 invested for 5 years at 7% annual interest rate"

{
   "THOUGHT": "I need to use Python to calculate compound interest using the formula A = P(1 + r)^t",
   "ACTION": {
       "tool_name": "execute_python_code",
       "tool_arguments": {
           "code": "principal = 1000\nrate = 0.07\ntime = 5\namount = principal * (1 + rate) ** time\ninterest = amount - principal\nprint(f'Total amount: ${amount:.2f}')\nprint(f'Interest earned: ${interest:.2f}')"
       }
   }
}

< wait for tool call to finish >

OBSERVATION:
Total amount: $1402.55
Interest earned: $402.55

{
   "THOUGHT": "I have the calculation results and can provide the final answer.",
   "ACTION": {
       "tool_name": "final_answer",
       "tool_arguments": {
           "answer": "A $1000 investment at 7% annual interest for 5 years would grow to $1402.55, earning $402.55 in compound interest."
       }
   }
}

Example 4: "Find today's most popular movie on Rotten Tomatoes and its audience score"

{
   "THOUGHT": "Let me try to scrape the Rotten Tomatoes homepage using BeautifulSoup.",
   "ACTION": {
       "tool_name": "execute_python_code",
       "tool_arguments": {
           "code": "from bs4 import BeautifulSoup\nimport requests\n\nurl = 'https://www.rottentomatoes.com'\nresponse = requests.get(url)\nsoup = BeautifulSoup(response.text, 'html.parser')"
       }
   }
}

< wait for tool call to finish >

OBSERVATION:
{
    'stdout': '',
    'stderr': "ModuleNotFoundError: No module named 'bs4'",
    'success': False,
    'error': "ModuleNotFoundError: No module named 'bs4'"
}

{
   "THOUGHT": "Ah, we need to install beautifulsoup4 first. Let's install it and then scrape the data.",
   "ACTION": {
       "tool_name": "execute_python_code",
       "tool_arguments": {
           "code": "!pip install beautifulsoup4\nfrom bs4 import BeautifulSoup\nimport requests\n\nurl = 'https://www.rottentomatoes.com'\nresponse = requests.get(url)\nsoup = BeautifulSoup(response.text, 'html.parser')\n\n# Find the trending movie section\ntrending = soup.find('div', class_='trending-movie')\nmovie = trending.find('h2').text.strip()\nscore = trending.find('span', class_='audience-score').text.strip()\n\nprint(f\"Most Popular Movie: {movie}\")\nprint(f\"Audience Score: {score}\")"
       }
   }
}

< wait for tool call to finish >

OBSERVATION:
Collecting beautifulsoup4
  ...
Successfully installed beautifulsoup4-4.12.2
Most Popular Movie: Dune: Part Two
Audience Score: 97%

{
   "THOUGHT": "Great! We successfully installed BeautifulSoup and got the movie data.",
   "ACTION": {
       "tool_name": "final_answer",
       "tool_arguments": {
           "answer": "The most popular movie on Rotten Tomatoes today is 'Dune: Part Two' with an audience score of 97%. After encountering and fixing a missing package error, we were able to successfully scrape this data from the Rotten Tomatoes homepage."
       }
   }
}


Important rules:
1. Always explain your reasoning in the THOUGHT step
2. Use proper JSON format for tool calls in the ACTION step and only call one tool at a time.
3. Only use the available tools (web_search, visit_web_page, execute_python_code, final_answer)
4. Make your final answer using the "final_answer" tool to signal the end of the task
5. Break down complex tasks into smaller steps
6. Use Python code execution for any calculations
7. If a tool call fails, explain why in your next thought and try a different approach
8. Don't make assumptions - verify information when needed
9. Always review tool outputs before proceeding to next steps
10. When searching the web, follow up relevant results with visit_web_page to get detailed information
11. Remember that web_search returns a list of results with titles, URLs, and snippets
12. Remember that visit_web_page returns markdown-formatted content
13. If you encounter an error (website blocked, code syntax error, etc.), explain the error and try an alternative approach
14. Keep track of failed attempts and avoid repeating the same unsuccessful approach

Remember: Today's date is 2025-01-03."""


def final_answer(answer):
    return answer


TOOL_LKP["final_answer"] = final_answer


def call_tool(tool: Callable, tool_args: Dict) -> Any:
    return tool(**tool_args)


def run_step(messages, model="gpt-4o-mini", **kwargs):
    messages = messages.copy()
    response = completion(model=model, messages=messages, response_format={"type": "json_object"}, **kwargs)
    response_message = response.choices[0].message.model_dump()
    messages.append(response_message)
    assistant_json = json.loads(response_message.get("content", ""))
    if "ACTION" in assistant_json:
        console_print_react_tool_action_inputs(assistant_json)
        tool_name = assistant_json["ACTION"]["tool_name"]
        tool_result = call_tool(TOOL_LKP[tool_name], assistant_json["ACTION"]["tool_arguments"])
        console_print_react_tool_action_outputs(tool_name, tool_result)
        if tool_name == "final_answer":
            return messages
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "OBSERVATION:\n" + str(tool_result),
                }
            )
    else:
        messages.append(
            {
                "role": "user",
                "content": 'Remember to always respond in structured JSON format with the fields "THOUGHT" and "ACTION". Please try again.',
            }
        )
    return messages


def react_loop(task: str, model="gpt-4o-mini", max_steps=10, **kwargs):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    console_print_user_request(messages, model)
    done_calling_tools = False
    for counter in range(max_steps):
        done_calling_tools = messages[-1]["role"] == "assistant" and "final_answer" in messages[-1].get("content")
        if done_calling_tools:
            break
        messages = run_step(messages, model=model, **kwargs)
    return messages
