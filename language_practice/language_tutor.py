import os, copy, json, time, pickle
import pandas as pd
from openai import OpenAI


api_key=open('openai_api_key.txt', 'r').read()
client = OpenAI(api_key=api_key)

# Big dataset of chinese words to learn and chinese_chars,pinyin_pronunciation,english_meaning
ALL_WORDS = [[p.strip() for p in l.split(',')] for l in open('chinese_words.txt').readlines()]
ch2pinyin = dict((p[0], p[1]) for p in ALL_WORDS)
ch2english = dict((p[0], p[2]) for p in ALL_WORDS)


N_KNOWN = 30
N_LEARNING = 5
N_TARGET_SENTENCES = 5
BATCH_SIZE = 20   # ask for several at once
N_SUCCESS_TO_MEMORIZE = 1  # this many successes in a row --> word memorized

'''
TODO
The API call slowed down a LOT when I switched to the current prompt.
It was much faster previously when I just asked for sentences
and figured out the target words and pronunciation myself.
'''

PROMPT_TEMPLATE = """
Here are a list of "Known words" that you can use as much as you want,
and a list of "Target words" - you can only use one of them in a sentence.
That idea is that every sentence should have exactly one "Target" word
and have all the other words be "Known" words.
You may ONLY use words from the "Known words" list and the "Target words" list.
Do NOT use any other words.
You may repeat words.

Known words:
{known_words_s}

Target words:
{target_words_s}

Task:
Write {n} idiomatic, natural, everyday Chinese sentences.
Each sentence must contain exactly one word from the "Target words" list
and have the rest be from the "Known words" list.

Each line should have the target word, the sentence (with spaces), pronunciation, and an English translation. 
Separated by pipes.

Example lines:
喜欢|我喜欢你|Wǒ xǐ huan nǐ|I like you
朋友|你是我的朋友|Nǐ shì wǒ de péng you|You are my friend
"""
def get_lines(known_words: list[list[str]], target_words: list[list[str]], n, model="gpt-5-mini") -> list[tuple[str,str]]:
    known_words_s = ','.join(known_words)
    target_words_s = ','.join(target_words)
    prompt = PROMPT_TEMPLATE.format(**locals())
    response = client.responses.create(
        model = model,
        input=prompt
    )
    lines = response.output_text.strip().splitlines()
    lines = [l.split('|') for l in lines if l.count('|')==3 and l.split('|')[0] in target_words]
    return lines

class LanguageTutor:
    def __init__(self):
        try:
            state = json.loads(open('state.json','r').read())
            print('Recovering saved state')
            assert False
        except:
            print('Initializing empty state')
            state = {
                'known_words': [wb[0] for wb in ALL_WORDS[:N_KNOWN]],
                'target_words': [wb[0] for wb in ALL_WORDS[N_KNOWN+1:N_KNOWN+N_LEARNING]],
                'perf': {w[0]:[] for w in ALL_WORDS}
            }
        self.state = state
    def save(self):
        open('state.json','w').write(json.dumps(self.state))
    def check(self):
        # Consistency of internal state
        known_words, target_words, perf = self.state['known_words'], self.state['target_words'], self.state['perf']
        assert not set(sknown_words).intersection(target_words)
    def get_sentences(self, n=5):
        known_words, target_words, perf = self.state['known_words'], self.state['target_words'], self.state['perf']
        return get_lines(known_words, target_words, N_TARGET_SENTENCES)
    def update(self, sentences, errors):
        known_words, target_words, perf = self.state['known_words'], self.state['target_words'], self.state['perf']
        new_target_words = []
        for i, (tw, s, pron, trans) in enumerate(sentences):
            if i in errors:
                perf[tw].append('fail')
            else:
                perf[tw].append('success')
            if len(perf[tw])>N_SUCCESS_TO_MEMORIZE and set(perf[tw][-1*N_SUCCESS_TO_MEMORIZE:])==set(['success']):
                assert not set(known_words).intersection(target_words), str(set(known_words).intersection(target_words))
                if tw in target_words:
                    known_words.append(tw)
                    target_words.remove(tw)
                    new_wb = [wb for wb in ALL_WORDS if wb[0] not in known_words and wb[0] not in target_words][0]
                    print('new_wb', new_wb)
                    target_words.append(new_wb[0])
                    new_target_words.append(new_wb[0])
        return new_target_words
