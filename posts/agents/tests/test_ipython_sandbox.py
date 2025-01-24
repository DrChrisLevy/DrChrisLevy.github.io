from typing import Any

from python_sandbox import create_sandbox, execute_python_code


def test_ipython_sandbox():
    sandbox = create_sandbox()

    codes_and_expected = [
        ("x = 42", {"stdout": "", "stderr": "", "success": True, "result": "None", "error": None}),
        ('print("hello")', {"stdout": "hello\n", "stderr": "", "success": True, "result": "None", "error": None}),
        ("42", {"stdout": "Out[1]: 42\n", "stderr": "", "success": True, "result": "42", "error": None}),
        ("x=x+x", {"stdout": "", "stderr": "", "success": True, "result": "None", "error": None}),
        ("print(x);x", {"stdout": "84\nOut[1]: 84\n", "stderr": "", "success": True, "result": "84", "error": None}),
        ("y", {"stdout": Any, "stderr": "", "success": False, "result": None, "error": "name 'y' is not defined"}),
        ("1/0", {"stdout": Any, "stderr": "", "success": False, "result": None, "error": "division by zero"}),
        (
            'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial(5)\nprint(f"Factorial: {result}")\nresult',
            {"stdout": "Factorial: 120\nOut[1]: 120\n", "stderr": "", "success": True, "result": "120", "error": None},
        ),
        (
            "\nclass Person:\n    def __init__(self, name):\n        self.name = name\n    \n    def greet(self):\n        return f'Hello, {self.name}!'\n",
            {"stdout": "", "stderr": "", "success": True, "result": "None", "error": None},
        ),
        (
            'person = Person("Alice"); person.greet()',
            {"stdout": "Out[1]: 'Hello, Alice!'\n", "stderr": "", "success": True, "result": "'Hello, Alice!'", "error": None},
        ),
        (
            "\ndef gen():\n    yield 1\n    yield 2\n    yield 3\n\ng = gen()\nnext(g)",
            {"stdout": "Out[1]: 1\n", "stderr": "", "success": True, "result": "1", "error": None},
        ),
        (
            '\nimport signal\n\ndef handler(signum, frame):\n    raise TimeoutError("Code execution timed out")\n\nsignal.signal(signal.SIGALRM, handler)\nsignal.alarm(1)  # Set timeout for 1 second\n\ni = 0\nwhile True:\n    i += 1',
            {"stdout": Any, "stderr": "", "success": False, "result": None, "error": "Code execution timed out"},
        ),
    ]
    try:
        for code, expected in codes_and_expected:
            result = execute_python_code(code, sandbox)
            # Handle special case for stdout when Any is expected
            if expected["stdout"] is Any:
                assert isinstance(result["stdout"], str)  # Just verify it's a string
                # Create new dict without stdout for remaining comparison
                result_no_stdout = {k: v for k, v in result.items() if k != "stdout"}
                expected_no_stdout = {k: v for k, v in expected.items() if k != "stdout"}
                assert result_no_stdout == expected_no_stdout
            else:
                assert result == expected
    except Exception as e:
        print(e)
        raise e
    finally:
        sandbox.terminate()


def test_pandas_sandbox():
    try:
        sandbox = create_sandbox()

        code = """
        import pandas as pd
        """
        observation = execute_python_code(code, sandbox)
        print(observation)

        code = """
        !pip install pandas
        import pandas as pd
        print("Pandas successfully imported!")
        """
        observation = execute_python_code(code, sandbox)
        print(observation)

        code = """
        # Create sample data to work with
        data = {
            'title': ['The Matrix', 'Inception', 'Jurassic Park', 'The Notebook', 'Die Hard'],
            'genre': ['Sci-Fi,Action', 'Sci-Fi,Thriller', 'Action,Adventure', 'Romance,Drama', 'Action,Thriller'],
            'rating': [8.7, 8.8, 8.1, 7.8, 8.2]
        }
        df = pd.DataFrame(data)
        print("Data sample:")
        print(df.head())
        """
        observation = execute_python_code(code, sandbox)
        print(observation)

        code = """
        # First, let's see what unique genres we have
        all_genres = set()
        for genres in df['genre']:
            all_genres.update(genres.split(','))
        print("Unique genres:", sorted(all_genres))
        """
        observation = execute_python_code(code, sandbox)
        print(observation)

        code = """
        def get_genre_stats(genre):
            # Get movies in this genre
            genre_movies = df[df['genre'].str.contains(genre)]
            avg_rating = genre_movies['rating'].mean()
            count = len(genre_movies)
            return {'genre': genre, 'avg_rating': avg_rating, 'movie_count': count}

        # Test with one genre
        action_stats = get_genre_stats('Action')
        print("Action genre stats:", action_stats)
        """
        observation = execute_python_code(code, sandbox)
        print(observation)

        code = """
        # Create stats for all genres
        genre_stats = [get_genre_stats(genre) for genre in all_genres]
        stats_df = pd.DataFrame(genre_stats)
        stats_df = stats_df.sort_values('avg_rating', ascending=False)
        print("\\nGenre Statistics:")
        print(stats_df)
        """
        observation = execute_python_code(code, sandbox)
        print(observation)
        assert observation == {
            "stdout": "\nGenre Statistics:\n       genre  avg_rating  movie_count\n0     Sci-Fi    8.750000            2\n4   Thriller    8.500000            2\n2     Action    8.333333            3\n1  Adventure    8.100000            1\n3    Romance    7.800000            1\n5      Drama    7.800000            1\n",
            "stderr": "",
            "success": True,
            "result": "None",
            "error": None,
        }
    except Exception as e:
        print(e)
        raise e
    finally:
        sandbox.terminate()
