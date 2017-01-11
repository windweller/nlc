"""
Preprocess ptb.train.txt
into train.x.txt and train.y.txt type
that can be load into NLCEnv

We also prepare 10 characters and 30 characters version
like in AC paper

"""

import os
import numpy as np
import string
import random
import re

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    # used by NLCEnv
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def char_tokenizer(sentence):
    # used by NLCEnv
    return list(sentence.strip())


def create_truncated_seq_ptb(keeplen=10, noise=0.3, filename="train"):
    """

    Parameters
    ----------
    keeplen: the number of characters we want to save, choices: 10, 30
    noise: 0.3, 0.5

    Returns
    -------
    """
    xs = []
    ys = []
    with open("./ptb."+filename+".txt", "r") as f:
        lines = f.readlines()
        # x = np.array(char_tokenizer(line)[:keeplen])
        # # 0 is drop
        # mask = np.random.choice([0,1], size=keeplen, p=[noise, 1-noise])
        # np.multiply(x, mask)
        for line in lines:
            line = line.replace("<unk>", "U")
            x = char_tokenizer(line)[:keeplen]
            y = char_tokenizer(line)[:keeplen]
            mask = np.random.choice([0, 1], size=keeplen, p=[noise, 1 - noise])
            for i in xrange(len(x)):
                if mask[i] == 0 and x[i] != " " and x[i] != "U":  # not sapce, not <unk>
                    # should we include special chars or entire vocab? eh
                    x[i] = random.choice(string.lowercase)

            xs.append(x)
            ys.append(y)

    with open("./" + filename + ".x.txt", "w") as f1:
        with open("./" + filename + ".y.txt", "w") as f2:
            for i in range(len(xs)):
                f1.write("".join(xs[i]) + "\n")
                f2.write("".join(ys[i]) + "\n")


if __name__ == '__main__':
    create_truncated_seq_ptb(keeplen=30, filename="valid")
