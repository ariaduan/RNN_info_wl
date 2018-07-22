This is the journal for the informativeness-wordlength relationship project.

We first used the pre-trained RNN model--"hidden650_batch128_dropout0.2_lr20.0.pt" to compute the information value of each word in a natural corpus--1B Word Benchmark Dataset (https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark to get data). The information value is computed according to the formula given in [Word lengths are optimized for efficient communication](http://www.pnas.org/content/pnas/108/9/3526.full.pdf). For the develop set, We extracted 30,000 sentences from the held-out set and computed the infomation value for each word within the dev set.

We also set specification ccriteria for the words we will finally use to test the relationship between informativeness and word length. That is we only consider words: 1)in lower case, 2)alphabetical, 3)appear more then mini-count times in the test set.

Finally, we did a plot comparison among word lists with different mini-count limit(mini-count = 0, 3, 5). 

