# EXPERT SYSTEM

This is an application for course IDSI31421.
It analysis overall sentiments of tweets between "Negative", "Neutral" and "Positive".

# Dataset

The dataset was taken from [Sentiment140](http://help.sentiment140.com/).
It contains 1.6 millions of tweets annotated with their 
polarity of the tweet (0 = negative, 2 = neutral, 4 = positive).
The dataset was automatically generated.
The generation algorithm considers for example
that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. 
Therefore, the dataset contains annotation errors and the classifier must be robust to noise.

Actually, the dataset contains very few neutral tweets (less than 200). So, we decided to
complete another dataset annotated by ourselves. Only 500 additional tweets were annotated.
The complementary dataset was formed with 5 different topics: "I", "global warming", "trump", "feminism", and "vegan".

# Code

The code is loosely inspired by the Flask application [mlsite](https://github.com/khushmeeet/mlsite/tree/master/app). 
This application provided a skeleton for the MVC architecture and the connection to Twitter.

However, many improvements were done:

- classifiers were trained from scratch using Sci-Kit learn and Sentiment140;
- the website was in Hindi and was poorly designed;
- some parts were also rewritten, e.g. Twitter authentification tokens were directly injected inside the code.


# Installation

The code was run with Python 3.6.8. One could use [`pyenv`](https://github.com/pyenv/pyenv) for set up a version locally:

``` 
    pyenv install 3.6.8
    pyenv local 3.6.8
```

Dependencies can then be installed with `requirements.txt` as in:

```
    pip install -r requirements
```

# Starting a Flask server

A server can be launched with:

```
     python run.py
```

Then, use your browser to go to [http://127.0.0.1:8000](http://127.0.0.1:8000).
