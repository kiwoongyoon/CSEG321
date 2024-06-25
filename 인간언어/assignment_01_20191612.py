from nltk.tokenize import word_tokenize
import argparse
import math
from collections import Counter
def bp_func(ref_tokens, gen_tokens):
    """
    short penalty 계산하는 함수 
    """
    ref_length,gen_length = len(ref_tokens),len(gen_tokens)
    if gen_length > ref_length:        return 1
    else:        return math.exp(1 - ref_length / gen_length) if gen_length > 0 else 0


def get_file_names():
    parser = argparse.ArgumentParser(description="파일 이름을 인자로 받아 처리하는 스크립트")
    parser.add_argument("rand_ref", help="첫 번째 텍스트 파일 이름")
    parser.add_argument("rand_gen", help="두 번째 텍스트 파일 이름")
    args = parser.parse_args()
    return args.rand_ref, args.rand_gen

def get_sentences(ref_path, gener_path):
    with open(ref_path, 'r', encoding='utf-8') as file:
        reference_sentences = file.read().strip().split('\n')  # 줄바꿈 문자 기준으로 나눔
    with open(gener_path, 'r', encoding='utf-8') as file:
        gen_sentences = file.read().strip().split('\n')  # 줄바꿈 문자 기준으로 나눔
    return reference_sentences, gen_sentences


def get_tokens(ref_sen, gen_sen):
    ref_token, gen_token = word_tokenize(ref_sen),word_tokenize(gen_sen)
    return ref_token,gen_token
    
def find_precision(reference_tokens, gen_tokens, n):
    gen_ngrams,reference_ngrams = make_ngram(gen_tokens, n),make_ngram(reference_tokens, n)
    ref_ngram_counts,gen_ngram_counts= Counter(reference_ngrams),Counter(gen_ngrams)
    correct_ngrams = sum(min(count, ref_ngram_counts[ngram]) for ngram, count in gen_ngram_counts.items())
    total_ngrams = len(gen_ngrams)
    precision = correct_ngrams / total_ngrams if total_ngrams > 0 else 0
    return precision
def rouge2(reference_tokens, candidate_tokens):
    """
    rouge-2 를 계산하는 함수 
    """
    candidate_ngrams = make_ngram(candidate_tokens, 2)
    reference_ngrams = make_ngram(reference_tokens, 2)
    ref_ngram_set,cand_ngram_set= set(reference_ngrams),set(candidate_ngrams)
    overlap_ngrams = cand_ngram_set.intersection(ref_ngram_set)
    rouge_score = len(overlap_ngrams) / len(ref_ngram_set) if len(ref_ngram_set) > 0 else 0
    return round(rouge_score, 4)

def make_ngram(tokens, n):
    ngrams_list = []
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        ngrams_list.append(" ".join(ngram))
    return ngrams_list

def bleu(ref_tokens, gen_tokens):
    """
    bleu score 위한 정확도 계산하기 
    """
    precisions = []
    for n in range(1, 5):
        precision = find_precision(ref_tokens, gen_tokens, n)
        precisions.append(round(precision,8))
    return precisions

def find_lcs(a,b):
    m,n = len(a),len(b)
    dp = [[0]*(n+1)for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if a[i-1]== b[j-1]:     dp[i][j] =dp[i-1][j-1]+1 
            else :    dp[i][j]= max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(reference_tokens, gen_tokens):
    """
     rouge_L을 계산하는 함수 
    """
    lcs_length = find_lcs(reference_tokens, gen_tokens)
    precision = lcs_length / len(gen_tokens) if len(gen_tokens) > 0 else 0
    recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
    if precision + recall == 0:    f1_score = 0
    else:    f1_score = (2 * precision * recall) / (precision + recall)
    return round(f1_score, 4)


a,b = get_file_names()
ref_sent,gen_sent = get_sentences(a,b)

bleu_scores= []
ro2_scores = []
rol_scores= []
for i in range(len(ref_sent)):
    ref_token,gen_token  = get_tokens(ref_sent[i], gen_sent[i])
    precisions = bleu(ref_token, gen_token)
    all_mul = 1 
    for j in range(len(precisions)):        all_mul *=precisions[j] 
    
    bp = bp_func(ref_token,gen_token)
    bleuscore = bp*(all_mul**(0.25))
    bleu_scores.append(bp*(all_mul**(0.25)))
    rouge_score = rouge2(ref_token, gen_token)
    ro2_scores.append(rouge2(ref_token, gen_token))
    rolscore = rouge_l(ref_token,gen_token)
    rol_scores.append(rouge_l(ref_token,gen_token))

print(f"Results - BLEU: {round(sum(bleu_scores)/len(ref_sent), 4)} ROUGE-2: {round(sum(ro2_scores)/len(ref_sent),4)} ROUGE-L: {round(sum(rol_scores)/len(ref_sent), 4)}")