import nltk
import os
import sys
import re
import itertools
import collections
import pkg_resources
from io import open
from corenlp import CoreNLPClient

class Rouge:
    DEFAULT_METRICS = {"rouge-n"}
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n", "rouge-l", "rouge-w"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {'words', 'bytes'}
    REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the official ROUGE script
    KEEP_CANNOT_IN_ONE_WORD = re.compile('cannot')
    KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile('_cannot_')

    WORDNET_KEY_VALUE = {}
    WORDNET_DB_FILEPATH = 'wordnet_key_value.txt'
    WORDNET_DB_FILEPATH_SPECIAL_CASE = 'wordnet_key_value_special_cases.txt'
    WORDNET_DB_DELIMITER = '|'
    STEMMER = None

    def __init__(self, metrics=None, max_n=None, limit_length=True, length_limit=665, length_limit_type='bytes', apply_avg=True, apply_best=False, stemming=True, alpha=0.5, weight_factor=1.0, ensure_compatibility=True):
        """
        Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is not high enough (as the average depends on this).
        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is slightly different and the Porter one implemented in NLTK. However, special cases of DUC 2004 have been traited.
                The solution would be to rewrite the whole perl stemming in python from the original script

        Args:
          metrics: What ROUGE score to compute. Available: ROUGE-N, ROUGE-L, ROUGE-W. Default: ROUGE-N
          max_n: N-grams for ROUGE-N if specify. Default:1
          limit_length: If the summaries must be truncated. Defaut:True
          length_limit: Number of the truncation where the unit is express int length_limit_Type. Default:665 (bytes)
          length_limit_type: Unit of length_limit. Available: words, bytes. Default: 'bytes'
          apply_avg: If we should average the score of multiple samples. Default: True. If apply_Avg & apply_best = False, then each ROUGE scores are independant
          apply_best: Take the best instead of the average. Default: False, then each ROUGE scores are independant
          stemming: Apply stemming to summaries. Default: True
          alpha: Alpha use to compute f1 score: P*R/((1-a)*P + a*R). Default:0.5
          weight_factor: Weight factor to be used for ROUGE-W. Official rouge score defines it at 1.2. Default: 1.0
          ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough). Default:True

        Raises:
          ValueError: raises exception if metric is not among AVAILABLE_METRICS
          ValueError: raises exception if length_limit_type is not among AVAILABLE_LENGTH_LIMIT_TYPES
          ValueError: raises exception if weight_factor < 0
        """
        self.metrics = metrics[:] if metrics is not None else Rouge.DEFAULT_METRICS
        for m in self.metrics:
            if m not in Rouge.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        # Add all rouge-n metrics
        if self.max_n is not None:
            index_rouge_n = self.metrics.index('rouge-n')
            del self.metrics[index_rouge_n]
            self.metrics += ['rouge-{}'.format(n) for n in range(1, self.max_n + 1)]
        self.metrics = set(self.metrics)

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in Rouge.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError("Unknown length_limit_type '{}'".format(length_limit_type))

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.limit_length = False
        self.length_limit_type = length_limit_type
        self.stemming = stemming

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.weight_factor = weight_factor
        if self.weight_factor <= 0:
            raise ValueError("ROUGE-W weight factor must greater than 0.")
        self.ensure_compatibility = ensure_compatibility

        # Load static objects
        if len(Rouge.WORDNET_KEY_VALUE) == 0:
            Rouge.load_wordnet_db(ensure_compatibility)
        if Rouge.STEMMER is None:
            Rouge.load_stemmer(ensure_compatibility)

        # Tokenizer
        self.tok_client = CoreNLPClient(annotators=['ssplit', 'tokenize'], stdout=sys.stderr, max_char_length=1020000)

    def update_ref(self, ref):
        """Updates the reference, including stemming and ngram computation."""
        self.ref_sents, self.ref_words = self.process_text(ref)
        self.ref_unigrams = self._get_ngrams(1, self.ref_words)
        self.ref_bigrams = self._get_ngrams(2, self.ref_words)


    def update_article(self, article):
        """Updates the article, including stemming and ngram computation."""
        self.article_sents, self.article_words = self.process_text(article)
        self.art_unigrams = [self._get_ngrams(1, sent) for sent in self.article_sents] # List of dicts
        self.art_bigrams = [self._get_ngrams(2, sent) for sent in self.article_sents] # List of dicts

    def update_hyp(self, summary_index):
        """Updates the current hypothesis"""
        self.hyp_sents = [self.article_sents[i] for i in summary_index]
        self.hyp_words = [word for sent in self.hyp_sents for word in sent]
        self.hyp_unigrams = Rouge.merge_ngram_set([self.art_unigrams[i] for i in summary_index])
        self.hyp_bigrams = Rouge.merge_ngram_set([self.art_bigrams[i] for i in summary_index])

    @staticmethod
    def merge_ngram_set(ngram_dicts):
        if len(ngram_dicts) == 1:
            return ngram_dicts[0]
        else:
            s1, s2, rest = ngram_dicts[0], ngram_dicts[1], ngram_dicts[2:]
            for key, val in s2.values():
                s1[key] += val
            return Rouge.merge_ngram_set([s1] + rest)

    def process_text(self, text):
        ann = self.tok_client.annotate(text)
        text_sents = []
        for i in range(len(ann.sentence)):
            ann_text = [ann.sentence[i].token[j].word for j in range(len(ann.sentence[i].token))]
            ann_text = Rouge.strip_punc(ann_text)
            if self.stemming:
                self.stem_tokens(ann_text)
            text_sents.append(ann_text)
        text_words = [word for sent in text_sents for word in sent]  # Flattened version
        return text_sents, text_words

    def stop(self):
        self.tok_client.stop()

    @staticmethod
    def strip_punc(tokens):
        return [t for t in tokens if not Rouge.REMOVE_CHAR_PATTERN.match(t)]

    @staticmethod
    def load_stemmer(ensure_compatibility):
        """
        Load the stemmer that is going to be used if stemming is enabled
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)
        """
        Rouge.STEMMER = nltk.stem.porter.PorterStemmer('ORIGINAL_ALGORITHM') if ensure_compatibility else nltk.stem.porter.PorterStemmer()

    @staticmethod
    def load_wordnet_db(ensure_compatibility):
        """
        Load WordNet database to apply specific rules instead of stemming + load file for special cases to ensure kind of compatibility (at list with DUC 2004) with the original stemmer used in the Perl script
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)

        Raises:
            FileNotFoundError: If one of both databases is not found
        """
        files_to_load = [Rouge.WORDNET_DB_FILEPATH]
        if ensure_compatibility:
            files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

        for wordnet_db in files_to_load:
            filepath = pkg_resources.resource_filename(__name__, wordnet_db)
            if not os.path.exists(filepath):
                raise FileNotFoundError("The file '{}' does not exist".format(filepath))

            with open(filepath, 'r', encoding='utf-8') as fp:
                for line in fp:
                    k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                    assert k not in Rouge.WORDNET_KEY_VALUE
                    Rouge.WORDNET_KEY_VALUE[k] = v

    @staticmethod
    def tokenize_text(text, language='english'):
        """
        Tokenize text in the specific language

        Args:
          text: The string text to tokenize
          language: Language of the text

        Returns:
          List of tokens of text
        """
        return nltk.word_tokenize(text, language)

    @staticmethod
    def split_into_sentences(text, ensure_compatibility, language='english'):
        """
        Split text into sentences, using specified language. Use PunktSentenceTokenizer

        Args:
          text: The string text to tokenize
          ensure_compatibility: Split sentences by '\n' instead of NLTK sentence tokenizer model
          language: Language of the text

        Returns:
          List of tokens of text
        """
        if ensure_compatibility:
            return text.split('\n')
        else:
            return nltk.sent_tokenize(text, language)

    @staticmethod
    def stem_tokens(tokens):
        """
        Apply WordNetDB rules or Stem each token of tokens

        Args:
          tokens: List of tokens to apply WordNetDB rules or to stem

        Returns:
          List of final stems
        """
        # Stemming & Wordnet apply only if token has at least 3 chars
        for i, token in enumerate(tokens):
            if len(token) > 0:
                if len(token) > 3:
                    if token in Rouge.WORDNET_KEY_VALUE:
                        token = Rouge.WORDNET_KEY_VALUE[token]
                    else:
                        token = Rouge.STEMMER.stem(token)
                    tokens[i] = token
        return tokens

    def _get_ngrams(self, n, use_ref):
        """
        Calcualtes n-grams.

        Args:
          n: which n-grams to calculate
          use_ref: Use reference or hypothesis

        Returns:
          A set of n-grams with their number of occurences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if use_ref:
            if n == 1:
                return self.ref_unigrams
            else:
                return self.ref_bigrams
        else:
            if n == 1:
                return self.ref_unigrams
            else:
                return self.ref_bigrams

        # ngram_set = collections.defaultdict(int)
        # max_index_ngram_start = len(text) - n
        # for i in range(max_index_ngram_start + 1):
        #     ngram_set[tuple(text[i:i + n])] += 1
        # return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        """
        Splits multiple sentences into words and flattens the result

        Args:
          sentences: list of string

        Returns:
          A list of words (split by white space)
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        return list(itertools.chain(*[_.split() for _ in sentences]))


    def _get_word_ngrams_and_length(self, n, use_ref):
        """
        Calculates word n-grams for multiple sentences.

        Args:
          n: wich n-grams to calculate
          use_ref: Use reference or hypothesis

        Returns:
          A set of n-grams, their frequency and #n-grams in sentences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        assert n == 1 or n == 2

        if use_ref:
            tokens = self.ref_words
        else:
            tokens = self.hyp_words
        return self._get_ngrams(n, use_ref), tokens, len(tokens) - (n - 1)

    def _get_unigrams(self, use_ref=True):
        """
        Calcualtes uni-grams.

        Args:
          use_ref: Use reference or hypothesis

        Returns:
          A set of n-grams and their freqneucy
        """
        if use_ref:
            return self.ref_unigrams, len(self.ref_words)
        else:
            return self.hyp_unigrams, len(self.hyp_words)

        # tokens = Rouge._split_into_words(sentences)
        # unigram_set = collections.defaultdict(int)
        # for token in tokens:
        #     unigram_set[token] += 1
        # return unigram_set, len(tokens)

    @staticmethod
    def _compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=0.5, weight_factor=1.0):
        """
        Compute precision, recall and f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          evaluated_count: #n-grams in the hypothesis
          reference_count: #n-grams in the reference
          overlapping_count: #n-grams in common between hypothesis and reference
          alpha: Value to use for the F1 score (default: 0.5)
          weight_factor: Weight factor if we have use ROUGE-W (default: 1.0, no impact)

        Returns:
          A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
        precision = 0.0 if evaluated_count == 0 else overlapping_count / float(evaluated_count)
        if weight_factor != 1.0:
            precision = precision ** (1.0 / weight_factor)
        recall = 0.0 if reference_count == 0 else overlapping_count / float(reference_count)
        if weight_factor != 1.0:
            recall = recall ** (1.0 / weight_factor)
        f1_score = Rouge._compute_f_score(precision, recall, alpha)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_f_score(precision, recall, alpha=0.5):
        """
        Compute f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          precision: precision
          recall: recall

        Returns:
            f1 score
        """
        return 0.0 if (recall == 0.0 or precision == 0.0) else precision * recall / ((1 - alpha) * precision + alpha * recall)

    def _compute_ngrams(self, n):
        """
        Computes n-grams overlap of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf

        Args:
          n: Size of ngram

        Returns:
          Number of n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times

        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if len(self.hyp_sents) <= 0 or len(self.ref_sents) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, _, evaluated_count = self._get_word_ngrams_and_length(n, use_ref=False)
        reference_ngrams, _, reference_count = self._get_word_ngrams_and_length(n, use_ref=True)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    def _compute_ngrams_lcs(self, weight_factor=1.0):
        """
        Computes ROUGE-L (summary level) of two text collections of sentences.
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Args:
          weight_factor: Weight factor to be used for WLCS (1.0 by default if LCS)
        Returns:
          Number of LCS n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times
        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = '|'
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = '^'
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = '<'

            return vals, dirs

        def _wlcs(x, y, weight_factor):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(float)
            dirs = collections.defaultdict(int)
            lengths = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        length_tmp = lengths[i - 1, j - 1]
                        vals[i, j] = vals[i - 1, j - 1] + (length_tmp + 1) ** weight_factor - length_tmp ** weight_factor
                        dirs[i, j] = '|'
                        lengths[i, j] = length_tmp + 1
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = '^'
                        lengths[i, j] = 0
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = '<'
                        lengths[i, j] = 0

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == '|':
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == '^':
                    m -= 1
                elif dirs[m, n] == '<':
                    n -= 1
                else:
                    raise UnboundLocalError('Illegal move')

            return mask

        if len(self.hyp_sents) <= 0 or len(self.ref_sents) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_unigrams_dict, evaluated_count = self._get_unigrams(use_ref=False)
        reference_unigrams_dict, reference_count = self._get_unigrams(use_ref=True)

        # Has to use weight factor for WLCS
        use_WLCS = weight_factor != 1.0
        if use_WLCS:
            evaluated_count = evaluated_count ** weight_factor
            reference_count = 0

        overlapping_count = 0.0
        for i in range(len(self.ref_sents)):
            reference_sentence_tokens = self.ref_sents[i]
            if use_WLCS:
                reference_count += len(reference_sentence_tokens) ** weight_factor
            hit_mask = [0 for _ in range(len(reference_sentence_tokens))]

            for i in range(len(self.hyp_sents)):
                evaluated_sentence_tokens = self.hyp_sents[i]

                if use_WLCS:
                    _, lcs_dirs = _wlcs(reference_sentence_tokens, evaluated_sentence_tokens, weight_factor)
                else:
                    _, lcs_dirs = _lcs(reference_sentence_tokens, evaluated_sentence_tokens)
                _mark_lcs(hit_mask, lcs_dirs, len(reference_sentence_tokens), len(evaluated_sentence_tokens))

            overlapping_count_length = 0
            for ref_token_id, val in enumerate(hit_mask):
                if val == 1:
                    token = reference_sentence_tokens[ref_token_id]
                    if evaluated_unigrams_dict[token] > 0 and reference_unigrams_dict[token] > 0:
                        evaluated_unigrams_dict[token] -= 1
                        reference_unigrams_dict[ref_token_id] -= 1

                        if use_WLCS:
                            overlapping_count_length += 1
                            if (ref_token_id + 1 < len(hit_mask) and hit_mask[ref_token_id + 1] == 0) or ref_token_id + 1 == len(hit_mask):
                                overlapping_count += overlapping_count_length ** weight_factor
                                overlapping_count_length = 0
                        else:
                            overlapping_count += 1

        if use_WLCS:
            reference_count = reference_count ** weight_factor

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self):
        """
        Compute precision, recall and f1 score between hypothesis and references

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
        scores = {}
        has_rouge_n_metric = len([metric for metric in self.metrics if metric.split('-')[-1].isdigit()]) > 0
        if has_rouge_n_metric:
            scores.update(self._get_scores_rouge_n())

        has_rouge_l_metric = len([metric for metric in self.metrics if metric.split('-')[-1].lower() == 'l']) > 0
        if has_rouge_l_metric:
            scores.update(self._get_scores_rouge_l_or_w(use_w=False))

        has_rouge_w_metric = len([metric for metric in self.metrics if metric.split('-')[-1].lower() == 'w']) > 0
        if has_rouge_w_metric:
            scores.update(self._get_scores_rouge_l_or_w(use_w=True))

        return scores

    def _get_scores_rouge_n(self):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metrics = [metric for metric in self.metrics if metric.split('-')[-1].isdigit()]
        scores = {metric: {stat:0.0 for stat in Rouge.STATS} for metric in metrics}

        # Compute scores
        for metric in metrics:
            suffix = metric.split('-')[-1]
            n = int(suffix)

            # Aggregate
            if self.apply_avg:
                # average model
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0

                hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams(n)
                total_hypothesis_ngrams_count += hypothesis_count
                total_reference_ngrams_count += reference_count
                total_ngrams_overlapping_count += overlapping_ngrams

                score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha)

                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            else:
                # Best model
                if self.apply_best:
                    best_current_score = None

                    hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams(n)
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    if best_current_score is None or score['r'] > best_current_score['r']:
                        best_current_score = score

                    for stat in Rouge.STATS:
                        scores[metric][stat] += best_current_score[stat]
                # Keep all
                else:
                    hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams(n)
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    for stat in Rouge.STATS:
                        scores[metric][0][stat].append(score[stat])
        return scores

    def _get_scores_rouge_l_or_w(self, use_w=False):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          use_w: Rouge L or W

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metric = "rouge-w" if use_w else "rouge-l"
        scores = {metric: {stat:0.0 for stat in Rouge.STATS}}

        # Compute scores
        # Aggregate
        if self.apply_avg:
            # average model
            total_hypothesis_ngrams_count = 0
            total_reference_ngrams_count = 0
            total_ngrams_overlapping_count = 0

            hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams_lcs(self.weight_factor if use_w else 1.0)
            total_hypothesis_ngrams_count += hypothesis_count
            total_reference_ngrams_count += reference_count
            total_ngrams_overlapping_count += overlapping_ngrams

            score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha, self.weight_factor)

            for stat in Rouge.STATS:
                scores[metric][stat] += score[stat]
        else:
            # Best model
            if self.apply_best:
                best_current_score = None
                best_current_score_wlcs = None

                hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams_lcs(self.weight_factor if use_w else 1.0)
                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)

                if use_w:
                    reference_count_for_score = reference_count ** (1.0 / self.weight_factor)
                    overlapping_ngrams_for_score = overlapping_ngrams
                    score_wlcs = (overlapping_ngrams_for_score / reference_count_for_score) ** (1.0 / self.weight_factor)

                    if best_current_score_wlcs is None or score_wlcs > best_current_score_wlcs:
                        best_current_score = score
                        best_current_score_wlcs = score_wlcs
                else:
                    if best_current_score is None or score['r'] > best_current_score['r']:
                        best_current_score = score

                for stat in Rouge.STATS:
                    scores[metric][stat] += best_current_score[stat]
            # Keep all
            else:
                hypothesis_count, reference_count, overlapping_ngrams = self._compute_ngrams_lcs(self.weight_factor if use_w else 1.0)
                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)

                for stat in Rouge.STATS:
                    scores[metric][0][stat].append(score[stat])
        return scores

    def _preprocess_summary_as_a_whole(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary as a whole

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            # By words
            if self.length_limit_type == 'words':
                summary = ' '.join(sentences)
                all_tokens = summary.split() # Counting as in the perls script
                summary = ' '.join(all_tokens[:self.length_limit])

            # By bytes
            elif self.length_limit_type == 'bytes':
                summary = ''
                current_len = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)

                    if current_len + sentence_len < self.length_limit:
                        if current_len != 0:
                            summary += ' '
                        summary += sentence
                        current_len += sentence_len
                    else:
                        if current_len > 0:
                            summary += ' '
                        summary += sentence[:self.length_limit-current_len]
                        break
        else:
            summary = ' '.join(sentences)

        summary = Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower()).strip()

        # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and "can not" as "can not",
        # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
        if self.ensure_compatibility:
            tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', summary))
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary))

        if self.stemming:
            self.stem_tokens(tokens) # stemming in-place

        if self.ensure_compatibility:
            preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))]
        else:
            preprocessed_summary = [' '.join(tokens)]

        return preprocessed_summary

    def _preprocess_summary_per_sentence(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary by sentences

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            final_sentences = []
            current_len = 0
            # By words
            if self.length_limit_type == 'words':
                for sentence in sentences:
                    tokens = sentence.strip().split()
                    tokens_len = len(tokens)
                    if current_len + tokens_len < self.length_limit:
                        sentence = ' '.join(tokens)
                        final_sentences.append(sentence)
                        current_len += tokens_len
                    else:
                        sentence = ' '.join(tokens[:self.length_limit - current_len])
                        final_sentences.append(sentence)
                        break
            # By bytes
            elif self.length_limit_type == 'bytes':
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)
                    if current_len + sentence_len < self.length_limit:
                        final_sentences.append(sentence)
                        current_len += sentence_len
                    else:
                        sentence = sentence[:self.length_limit - current_len]
                        final_sentences.append(sentence)
                        break
            sentences = final_sentences

        final_sentences = []
        for sentence in sentences:
            sentence = Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence.lower()).strip()

            # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and "can not" as "can not",
            # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
            if self.ensure_compatibility:
                tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', sentence))
            else:
                tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence))

            if self.stemming:
                self.stem_tokens(tokens) # stemming in-place

            if self.ensure_compatibility:
                sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
            else:
                sentence = ' '.join(tokens)

            final_sentences.append(sentence)

        return final_sentences
