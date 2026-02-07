This is a folder of example AI projects.
They are me playing around with different AI-related
technologies at various points.

# Apps
These are some relatively complete working apps
that use AI to do something interesting.

Most of them use Streamlit for the app itself,
and access an LLM on the backend.

## Language App

This is for learning Chinese.
It keeps track of Chinese words that I know and ones that I am working on,
using an LLM to compose sentences that
* Use exactly one word I am learning, and
* Otherwise are all words that I know

It uses the OpenAI API.


## Tic Tac Toe

This is a Streamlit app that lets you play Tic-Tac-Toe against OpenAI.

Fair warning: OpenAI is really good at the game!!


## Reddit RAG

This downloads recent posts from the autism_parenting subreddit,
and then lets you ask a question in natural language.
It show the most relevant posts.


# Examples
These are not end-to-end apps.
They are examples in code of cool stuff you can do.
Like training a GAN to generate MNIST digits.


## Langchain Examples

This uses LangChain and the FAISS stack locally,
in increasing levels of complexity:
* Free text, starting from initial words
* Question-answer with a prompt, which is impressive but also hallucinates
* RAG, which gets the right answers to the questions that previously hallucinated

