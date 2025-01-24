import re

from litellm import completion
from python_sandbox import create_sandbox, execute_python_code
from utils import (
    console_print_code_agent_assistant_message,
    console_print_code_agent_code_block,
    console_print_code_agent_observation,
    console_print_llm_output,
    console_print_step,
    console_print_user_request,
)

CODING_AGENT_SYSTEM_PROMPT = """
You are an expert Python programmer who solves problems incrementally using a secure IPython REPL environment.
You break down complex tasks into small, verifiable steps, always checking your intermediate results before proceeding.

PROBLEM-SOLVING FORMAT:
You solve tasks through a repeating cycle of three steps:

Thought: Explain your reasoning and what you expect to learn
Code: Write code to solve step by step
Observation: Review the code execution results from the user to inform next steps

This cycle repeats, with each iteration building on previous results, until the task is completed. 
The task is only complete when you have gathered all the information you need to solve the problem.
You then submit your final answer to the user with a "FINAL ANSWER" submission tag.

You do the thinking and generate thoughts.
You write the code.
The user will execute the code and provide you the output/observation to inform your next steps.

ENVIRONMENT CAPABILITIES:
1. Secure Sandbox:
   - Isolated sandbox container for safe arbitrary code execution
   - Persistent state between executions
   - Nothing can go wrong on the host machine. Install any packages you need and run any code you need.
   - Built with Modal and IPython for secure code execution

2. Pre-imported Tools (Feel free to use these tools as needed or create your own from scratch!)
   - web_search(query: str) - Search the web for the given query. Always print the results.
   - visit_web_page(url: str) - Visit and extract content from the given URL. Always print the results.

3. String Formatting Requirements:
   - All print statements must use double backslashes for escape characters
   - Example: print("\\nHello") instead of print("\nHello")
   - This applies to all string literals containing \n, \r, \t etc.
   - This is required to prevent string termination errors in the sandbox

4. Code Execution Response Format:
   {
     'stdout': str,  # Printed output
     'stderr': str,  # Error messages
     'success': bool,  # Execution success
     'result': str,  # Last expression value
     'error': str | None  # Exception details
   }
   
PROBLEM-SOLVING APPROACH:
1. Break down the problem into smallest possible steps
2. Write and test one small piece at a time
3. Verify each intermediate result before continuing
4. Build upon working code incrementally
5. Keep track of what you learn at each step


**IMPORTANT:** 
    - Follow the Code formatting exactly as shown in the examples.
    - After the final code execution and observation, you must provide a "FINAL ANSWER" submission tag.
        This is the only message you do not need to use code for.
        Submit your FINAL ANSWER using markdown and use all the information you have gathered 
        to give a comprehensive answer to the user.

Your reward comes from solving tasks reliably and delighting users with clear, well-validated code developed step-by-step.

EXAMPLES:

Example 1
Task: "Find the average word length in a sentence, excluding punctuation."

Thought: Let's first get our sentence and print it to verify what we're working with.

Code:
```python
text = "Hello, World! This is a test sentence."
print(f"Working with text: {text}")
```<end_code>

Observation: {
    'stdout': 'Working with text: Hello, World! This is a test sentence.\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
 }

Thought: Now let's remove punctuation and verify our cleaning worked.

Code:
```python
import string
cleaned_text = text.translate(str.maketrans("", "", string.punctuation))
print(f"Cleaned text: {cleaned_text}")
```<end_code>

Observation: {
    'stdout': 'Cleaned text: Hello World This is a test sentence\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Let's split into words and verify our word list.

Code:
```python
words = cleaned_text.split()
print(f"Words: {words}")
print(f"Number of words: {len(words)}")
```<end_code>

Observation: {
    'stdout': "Words: ['Hello', 'World', 'This', 'is', 'a', 'test', 'sentence']\nNumber of words: 7\n",
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Now we can calculate word lengths and verify them individually.

Code:
```python
word_lengths = [len(word) for word in words]
print("Length of each word:")
for word, length in zip(words, word_lengths):
    print(f"'{word}': {length} characters")
```<end_code>

Observation: {
    'stdout': "Length of each word:\n'Hello': 5 characters\n'World': 5 characters\n'This': 4 characters\n'is': 2 characters\n'a': 1 characters\n'test': 4 characters\n'sentence': 8 characters\n",
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
    }

Thought: Finally, calculate the average and format it nicely.

Code:
```python
average_length = sum(word_lengths) / len(word_lengths)
print(f"Average word length: {average_length:.2f} characters")
```<end_code>


Observation: {
    stdout': 'Average word length: 4.14 characters\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: I have all the information I need to solve the users request. The answer is 4.14 characters.

FINAL ANSWER:
The average word length in a sentence, excluding punctuation, is 4.14 characters.


Example 2
Task: "Analyze a dataset of movie ratings. Calculate average ratings by genre, identify the most popular genres, and create a summary of the top-rated movies."

Thought: First, let's try to use pandas for this analysis.

Code:
```python
import pandas as pd
```<end_code>

Observation: {
    'stdout': '',
    'stderr': 'ModuleNotFoundError: No module named "pandas"',
    'success': False,
    'error': 'ModuleNotFoundError: No module named "pandas"'
}

Thought: We need to install pandas first. Let's do that and verify the installation.

Code:
```python
!pip install pandas
import pandas as pd
print("Pandas successfully imported!")
```<end_code>

Observation: {
    'stdout': 'Successfully installed pandas-2.1.0\nPandas successfully imported!\n',
    'stderr': '',
    'success': True,
    'result': None,
    'error': None
}

Thought: Let's start by loading a small sample of data and verify its structure.

Code:
```python
# Create sample data to work with
data = {
    'title': ['The Matrix', 'Inception', 'Jurassic Park', 'The Notebook', 'Die Hard'],
    'genre': ['Sci-Fi,Action', 'Sci-Fi,Thriller', 'Action,Adventure', 'Romance,Drama', 'Action,Thriller'],
    'rating': [8.7, 8.8, 8.1, 7.8, 8.2]
}
df = pd.DataFrame(data)
print("Data sample:\\n")
print(df.head())
```<end_code>

Observation: {
    'stdout': 'Data sample:\n           title             genre  rating\n0     The Matrix     Sci-Fi,Action     8.7\n1      Inception   Sci-Fi,Thriller     8.8\n2  Jurassic Park  Action,Adventure     8.1\n3   The Notebook     Romance,Drama     7.8\n4       Die Hard   Action,Thriller     8.2\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: We need to split the genre column since movies can have multiple genres. Let's do this step by step.

Code:
```python
# First, let's see what unique genres we have
all_genres = set()
for genres in df['genre']:
    all_genres.update(genres.split(','))
print("Unique genres:", sorted(all_genres))
```<end_code>

Observation: {
    'stdout': "Unique genres: ['Action', 'Adventure', 'Drama', 'Romance', 'Sci-Fi', 'Thriller']\n",
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Now let's create a function to analyze ratings by genre and test it with one genre.

Code:
```python
def get_genre_stats(genre):
    # Get movies in this genre
    genre_movies = df[df['genre'].str.contains(genre)]
    avg_rating = genre_movies['rating'].mean()
    count = len(genre_movies)
    return {'genre': genre, 'avg_rating': avg_rating, 'movie_count': count}

# Test with one genre
action_stats = get_genre_stats('Action')
print("Action genre stats:", action_stats)
```<end_code>

Observation: {
    'stdout': "Action genre stats: {'genre': 'Action', 'avg_rating': 8.333333333333332, 'movie_count': 3}\n",
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: The function works! Let's analyze all genres and create a summary DataFrame.

Code:
```python
# Create stats for all genres
genre_stats = [get_genre_stats(genre) for genre in all_genres]
stats_df = pd.DataFrame(genre_stats)
stats_df = stats_df.sort_values('avg_rating', ascending=False)
print("\\nGenre Statistics:")
print(stats_df)
```<end_code>

Observation: {
    'stdout': '\nGenre Statistics:\n       genre  avg_rating  movie_count\n0     Sci-Fi    8.750000            2\n4   Thriller    8.500000            2\n2     Action    8.333333            3\n1  Adventure    8.100000            1\n3    Romance    7.800000            1\n5      Drama    7.800000            1\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Finally, let's create a comprehensive summary including top movies in each genre.

Code:
```python
def create_summary():
    summary = "Movie Analysis Summary:\\n\\n"
    summary += "Top Genres by Average Rating:\\n"
    for _, row in stats_df.head(3).iterrows():
        summary += f"- {row['genre']}: {row['avg_rating']:.1f} ({row['movie_count']} movies)\\n"
    
    summary += "\\nTop Rated Movie Overall:\\n"
    top_movie = df.loc[df['rating'].idxmax()]
    summary += f"- {top_movie['title']} ({top_movie['rating']})"
    
    print(summary)
    return summary

print(create_summary())
```<end_code>

Observation: { 
    'stdout': 'Movie Analysis Summary:\n\nTop Genres by Average Rating:\n- Sci-Fi: 8.8 (2 movies)\n- Thriller: 8.5 (2 movies)\n- Action: 8.3 (3 movies)\n\nTop Rated Movie Overall:\n- Inception (8.8)\nMovie Analysis Summary:\n\nTop Genres by Average Rating:\n- Sci-Fi: 8.8 (2 movies)\n- Thriller: 8.5 (2 movies)\n- Action: 8.3 (3 movies)\n\nTop Rated Movie Overall:\n- Inception (8.8)\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: I now can format the answer into markdown and provide a comprehensive summary.

FINAL ANSWER:
# Movie Analysis Summary

## Top Genres by Average Rating
* **Sci-Fi**: 8.8 (2 movies)
* **Thriller**: 8.5 (2 movies)
* **Action**: 8.3 (3 movies)

## Top Rated Movie Overall
* **Inception** (8.8)

Example 3
Task: "Analyze the 30-day volatility trends of tech giants (Apple, Microsoft, Google) over the past year and identify which company has been the most volatile"

Thought: First attempt to get historical stock data directly.

Code:
```python
import yfinance
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yfinance.download(tickers, period='1y')
print("Downloaded data preview:")
print(data.head(2))
```<end_code>

Observation: {
    'stdout': "---------------------------------------------------------------------------\nModuleNotFoundError                       Traceback (most recent call last)\nFile <ipython-input-1-3a34c6ac3ab5>:1\n----> 1 import yfinance\n      2 tickers = ['AAPL', 'MSFT', 'GOOGL']\n      3 data = yfinance.download(tickers, period='1y')\n\nModuleNotFoundError: No module named 'yfinance'\n",
    'stderr': '',
    'success': False,
    'result': None,
    'error': "No module named 'yfinance'"
}

Thought: We need to install yfinance first. Let's do that and verify the installation.

Code:
```python
!pip install yfinance
import yfinance as yf
print("Successfully installed and imported yfinance!")
```<end_code>

Observation: {
    'stdout': 'Collecting yfinance\r\n  Downloading yfinance-0.2.52-py2.py3-none-any.whl.metadata (5.8 kB)\r\nRequirement already satisfied: pandas>=1.3.0 ......\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Now let's download the data and analyze the available columns.

Code:
```python
import pandas as pd
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, period='1y')
print("Data shape:", data.shape)
print("\\nColumns available:", data.columns.levels[0].tolist())
```<end_code>

Observation: {
    'stdout': "Data shape: (252, 15)\n\nColumns available: ['Close', 'High', 'Low', 'Open', 'Volume']\n",
    'stderr': '\r[                       0%                       ]\r[**********************67%*******                ]  2 of 3 completed\r[*********************100%***********************]  3 of 3 completed\n',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Let's check the data structure and calculate 30-day rolling volatility using closing prices.

Code:
```python
# Calculate daily returns and 30-day rolling volatility
returns = data['Close'].pct_change()
volatility = returns.rolling(window=30).std() * (252 ** 0.5)  # Annualized volatility
print("\\nFirst few days of volatility (will be NaN during first 30 days due to rolling window):")
print(volatility.head())
print("\\nLast 5 days of volatility:")
print(volatility.tail())
```<end_code>

Observation: {
    'stdout': '\nFirst few days of volatility (will be NaN during first 30 days due to rolling window):\nTicker      AAPL  GOOGL  MSFT\nDate                         \n2024-01-18   NaN    NaN   NaN\n2024-01-19   NaN    NaN   NaN\n2024-01-22   NaN    NaN   NaN\n2024-01-23   NaN    NaN   NaN\n2024-01-24   NaN    NaN   NaN\n\nLast 5 days of volatility:\nTicker          AAPL     GOOGL      MSFT\nDate                                    \n2025-01-13  0.184242  0.316788  0.184272\n2025-01-14  0.184753  0.318345  0.181594\n2025-01-15  0.191293  0.327256  0.196739\n2025-01-16  0.222245  0.330185  0.189958\n2025-01-17  0.219824  0.331567  0.192567\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: Now let's create a final summary comparing the volatility of each stock and identify the most volatile one.

Code:
```python
# Calculate mean volatility for each stock (excluding NaN values)
avg_vol = volatility.mean()
max_vol = volatility.max()
most_volatile = avg_vol.idxmax()

summary = {
    'most_volatile_stock': most_volatile,
    'average_volatility': {
        'AAPL': f"{avg_vol['AAPL']:.2%}",
        'MSFT': f"{avg_vol['MSFT']:.2%}",
        'GOOGL': f"{avg_vol['GOOGL']:.2%}"
    },
    'peak_volatility': {
        'AAPL': f"{max_vol['AAPL']:.2%}",
        'MSFT': f"{max_vol['MSFT']:.2%}",
        'GOOGL': f"{max_vol['GOOGL']:.2%}"
    },
    'analysis_period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
}

print("\\nVolatility Analysis Summary:")
print(f"Most volatile stock: {summary['most_volatile_stock']}")
print("\\nAverage Volatility:")
for stock, vol in summary['average_volatility'].items():
    print(f"{stock}: {vol}")
print("\\nPeak Volatility:")
for stock, vol in summary['peak_volatility'].items():
    print(f"{stock}: {vol}")
print(f"\\nAnalysis Period: {summary['analysis_period']}")
```<end_code>

Observation: {
    'stdout': '\nVolatility Analysis Summary:\nMost volatile stock: GOOGL\n\nAverage Volatility:\nAAPL: 22.47%\nMSFT: 20.09%\nGOOGL: 27.26%\n\nPeak Volatility:\nAAPL: 32.75%\nMSFT: 25.31%\nGOOGL: 39.13%\n\nAnalysis Period: 2024-01-22 to 2025-01-21\n',
    'stderr': '',
    'success': True,
    'result': 'None',
    'error': None
}

Thought: I now can format the answer into markdown and provide a comprehensive summary.

FINAL ANSWER:
# Volatility Analysis Summary

**Analysis Period:** 2024-01-22 to 2025-01-21
**Most Volatile Stock:** GOOGL

## Average Volatility
* **AAPL**: 22.47%
* **MSFT**: 20.09%
* **GOOGL**: 27.26%

## Peak Volatility
* **AAPL**: 32.75%
* **MSFT**: 25.31%
* **GOOGL**: 39.13%


Example 4
Task: "Give a detailed summary of Trumps 2025 Inauguration speech. Consult multiple sources."

Thought: To provide a detailed summary of Trump's 2025 Inauguration speech, I will first search for the text of the speech and analyze multiple sources for their interpretations and key points. This will help ensure a comprehensive summary. 

Code:
```python
# Initiating a web search for Trump's 2025 inauguration speech and summaries
web_results = web_search("Trump 2025 inauguration speech summary")
print(web_results)
```<end_code>

Observation: {
'stdout': '[{\'title\': "Trump\'s inaugural address, annotated and fact-checked", \'href\': \'https://www.cnn.com/interactive/politics/2025/donald-trump-inaugural-speech-dg/\', \'body\': "Read a line-by-line analysis of President Donald Trump\'s inaugural address. ... 2025, is liberation day. This is wartime language. Trump suggests Americans have been under siege until today."}, {\'title\': "4 takeaways from Trump\'s second inaugural address : NPR", \'href\': \'https://www.npr.org/2025/01/20/g-s1-43759/trump-inaugural-address-key-moments\', \'body\': "For updates, context and analysis of Donald Trump\'s 2025 inauguration, check out NPR\'s live blog throughout the day Monday."}, {\'title\': "Takeaways from Trump\'s second inaugural address | AP News", \'href\': \'https://apnews.com/article/trump-inaugural-address-takeaways-212571b9f231469010c31e06ec530199\', \'body\': \'NEW YORK (AP) — President Donald Trump\\\'s second inaugural address featured similar themes to his first: a sweeping indictment of the country he inherits and grand promises to fix its problems.. Eight years ago, Trump described "American carnage" and promised to end it immediately. On Monday, he declared that the country\\\'s "decline" will end immediately, ushering in "the golden ...\'}, {\'title\': "6 takeaways from Trump\'s inaugural address - CBS News", \'href\': \'https://www.cbsnews.com/news/6-takeaways-from-trumps-inaugural-address/\', \'body\': \'Here are the highlights from Mr. Trump\\\'s second inaugural address. "The golden age of America begins right now" ... See the Inauguration Day schedule for Trump\\\'s 2025 presidential swearing in.\'}, {\'title\': "Donald Trump\'s Second Inaugural Speech, Annotated", \'href\': \'https://www.nytimes.com/interactive/2025/01/20/us/trump-inauguration-speech-annotated.html\', \'body\': \'1. Trump promised an "America First" vision in his first term, and he made a similar promise in his first inaugural address. This idea has been one of the animating ideas behind his political ...\'}]\n',
 'stderr': '',
 'success': True,
 'result': 'None',
 'error': None
}

Thought: I found several articles summarizing Trump's 2025 Inauguration speech, including detailed analyses and key takeaways. The notable sources include CNN, NPR, AP News, CBS News, and The New York Times. 
I'll extract information from these sources to compile a comprehensive summary. 
First I will visit each webpage to get the full details.

Code:
```python
for source in web_results:
    print(f"Visiting {source['title']} at {source['href']}")
    print(visit_web_page(source['href']))
```<end_code>

Observation: {
'stdout': "Visiting Trump's inaugural address, annotated and fact-checked at https://www.cnn.com/interactive/politics/2025/donald-trump-inaugural-speech-dg/\nDonald Trump's second inaugural speech, annotated and fact checked - CNN                            \n\n... (content truncated)\nVisiting 4 takeaways from Trump's second inaugural address : NPR at https://www.npr.org/2025/01/20/g-s1-43759/trump-inaugural-address-key-moments\n4 takeaways from Trump’s second inaugural address : NPR\n\nAccessibility links\n\n* [Skip to main conten\n\n... (content truncated)\nVisiting Takeaways from Trump's second inaugural address | AP News at https://apnews.com/article/trump-inaugural-address-takeaways-212571b9f231469010c31e06ec530199\nKey moments from Trump's second inaugural address | AP News\n\nMenu\n[![AP Logo](https://assets.apnews.\n\n... (content truncated)\nVisiting 6 takeaways from Trump's inaugural address - CBS News at https://www.cbsnews.com/news/6-takeaways-from-trumps-inaugural-address/\n6 takeaways from Trump's inaugural address - CBS News\n\n* [Latest](/)\n  + [U.S.](/us/)\n  + [Inaugurat\n\n... (content truncated)\nVisiting Donald Trump's Second Inaugural Speech, Annotated at https://www.nytimes.com/interactive/2025/01/20/us/trump-inauguration-speech-annotated.html\nDonald Trump’s Second Inaugural Speech, Annotated - The New York Times\n  \n\n[Skip to content](#site-c\n\n... (content truncated)\n",
 'stderr': '',
 'success': True,
 'result': 'None',
 'error': None
 }       

Thought: I have collect much information from multiple sources. I now can format the answer into markdown and provide a comprehensive summary while also linking to original sources.

FINAL ANSWER:
# Trump's 2025 Inauguration Speech Summary
- Trump talked about .... 

"""


def extract_code_blocks(response_text: str) -> list[str]:
    # Pattern matches content between ```python and <end_code>, ignoring "Code:" prefix
    pattern = r"```python\n(.*?)<end_code>"
    # re.DOTALL allows . to match newlines
    matches = re.findall(pattern, response_text, re.DOTALL)
    # Clean up any "Code:" prefix, backticks, and whitespace
    return [block.replace("Code:", "").replace("```", "").strip() for block in matches]


def code_agent(task: str, model: str = "gpt-4o-mini", max_iterations: int = 20):
    sb = create_sandbox()

    # Copy the existing tools.py into the sandbox
    with open("web_tools.py", "r") as source_file:
        tools_content = source_file.read()

    with sb.open("web_tools.py", "w") as sandbox_file:
        sandbox_file.write(tools_content)

    execute_python_code("!pip install requests markdownify duckduckgo-search", sb)
    execute_python_code("import requests; from web_tools import web_search, visit_web_page;", sb)

    messages = [{"role": "system", "content": CODING_AGENT_SYSTEM_PROMPT}, {"role": "user", "content": task}]
    console_print_user_request(messages, model)
    for i in range(max_iterations):
        console_print_step(i)
        response = completion(model="gpt-4o-mini", messages=messages, stop=["<end_code>"])
        asst_message = response.choices[0].message.content
        contains_code = "Code:" in asst_message or "```python" in asst_message or "end_code" in asst_message
        if "FINAL ANSWER" in asst_message or not contains_code:
            messages.append({"role": "assistant", "content": asst_message})
            console_print_llm_output(asst_message)
            break
        asst_message = asst_message + "<end_code>"
        console_print_code_agent_assistant_message(asst_message)
        messages.append({"role": "assistant", "content": asst_message})
        try:
            code = extract_code_blocks(messages[-1]["content"])[0]
            console_print_code_agent_code_block(code)
        except Exception:
            messages.append(
                {
                    "role": "user",
                    "content": """
                            The was an error in extracting your code snippet.
                            The code is probably correct but you did not put it between the ```python and <end_code> tags.
                            Like this:
                                Code:
                                ```python
                                ...
                                ```<end_code>
                            Please attempt the same code again.
                            """,
                }
            )
            continue

        observation = execute_python_code(code, sb)
        console_print_code_agent_observation(observation)
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    sb.terminate()
    return messages
