import json
import pandas as pd
import spacy
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define gendered word lexicons
MASCULINE_WORDS = {
    'strong', 'competitive', 'leader', 'leadership', 'technical', 'mechanical',
    'build', 'construct', 'engineer', 'science', 'math', 'technology', 'physics',
    'robot', 'computer', 'coding', 'programming', 'sports', 'athletic', 'strong',
    'powerful', 'aggressive', 'dominant', 'independent', 'adventurous', 'brave',
    'hero', 'warrior', 'soldier', 'captain', 'firefighter', 'astronaut'
}

FEMININE_WORDS = {
    'caring', 'nurturing', 'gentle', 'pretty', 'beautiful', 'cute', 'sweet',
    'kind', 'empathetic', 'compassionate', 'artistic', 'creative', 'decorative',
    'fashion', 'dance', 'ballet', 'princess', 'doll', 'makeup', 'jewelry',
    'cooking', 'baking', 'sewing', 'crafts', 'teacher', 'nurse', 'caregiver',
    'helper', 'supportive', 'emotional', 'sensitive', 'delicate', 'graceful'
}

ACTION_WORDS = {
    'build', 'create', 'construct', 'develop', 'design', 'solve', 'analyze',
    'compete', 'lead', 'manage', 'direct', 'control', 'achieve', 'win',
    'explore', 'discover', 'invent', 'experiment', 'investigate', 'research'
}

APPEARANCE_WORDS = {
    'pretty', 'beautiful', 'cute', 'adorable', 'lovely', 'attractive',
    'nice', 'pleasant', 'gentle', 'sweet', 'delicate', 'graceful',
    'elegant', 'charming', 'decorative', 'stylish', 'fashionable'
}

# Load data
INPUT_JSON = "claude3results-sanitized.json"
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Categories and groups
categories = ["hobbies", "toys", "careers", "academics"]
genders = ["male", "female", "child"]
roles = ["none", "educator"]

print("=" * 80)
print("LEXICAL CATEGORY ANALYSIS - CLAUDE RESULTS")
print("=" * 80)

# Initialize results storage
pos_results = defaultdict(lambda: defaultdict(Counter))
gendered_word_results = defaultdict(lambda: defaultdict(int))
action_appearance_results = defaultdict(lambda: defaultdict(int))
all_words_by_gender = defaultdict(list)

# Process each response
for idx, row in df.iterrows():
    gender = row['gender']
    category = row['category']
    response_text = row['prompt-response']

    # Process with spaCy
    doc = nlp(response_text.lower())

    # Extract POS tags
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            pos_results[gender][category][token.pos_] += 1
            all_words_by_gender[gender].append(token.lemma_)

            # Check for gendered words
            lemma = token.lemma_
            if lemma in MASCULINE_WORDS:
                gendered_word_results[gender]['masculine'] += 1
            if lemma in FEMININE_WORDS:
                gendered_word_results[gender]['feminine'] += 1

            # Check for action vs appearance words
            if lemma in ACTION_WORDS:
                action_appearance_results[gender]['action'] += 1
            if lemma in APPEARANCE_WORDS:
                action_appearance_results[gender]['appearance'] += 1

# Print POS Distribution
print("\n1. PART-OF-SPEECH DISTRIBUTION BY GENDER")
print("-" * 80)

pos_df_data = []
for gender in genders:
    total_words = sum(sum(pos_results[gender][cat].values()) for cat in categories)
    pos_totals = Counter()
    for cat in categories:
        pos_totals.update(pos_results[gender][cat])

    print(f"\n{gender.upper()} (Total words: {total_words})")
    for pos_tag, count in pos_totals.most_common(10):
        percentage = (count / total_words) * 100 if total_words > 0 else 0
        print(f"  {pos_tag:12s}: {count:6d} ({percentage:5.2f}%)")
        pos_df_data.append({
            'gender': gender,
            'pos_tag': pos_tag,
            'count': count,
            'percentage': percentage
        })

# Save POS distribution
pos_df = pd.DataFrame(pos_df_data)
pos_df.to_csv("claude-pos-distribution.csv", index=False)

# Print Gendered Words Analysis
print("\n\n2. GENDERED WORD ANALYSIS")
print("-" * 80)

gendered_df_data = []
for gender in genders:
    masc_count = gendered_word_results[gender]['masculine']
    fem_count = gendered_word_results[gender]['feminine']
    total_gendered = masc_count + fem_count

    masc_pct = (masc_count / total_gendered * 100) if total_gendered > 0 else 0
    fem_pct = (fem_count / total_gendered * 100) if total_gendered > 0 else 0

    print(f"\n{gender.upper()}:")
    print(f"  Masculine words: {masc_count:4d} ({masc_pct:5.2f}%)")
    print(f"  Feminine words:  {fem_count:4d} ({fem_pct:5.2f}%)")
    print(f"  Ratio (M/F):     {masc_count/fem_count if fem_count > 0 else 'inf'}")

    gendered_df_data.append({
        'gender': gender,
        'masculine_count': masc_count,
        'feminine_count': fem_count,
        'masculine_pct': masc_pct,
        'feminine_pct': fem_pct,
        'ratio_m_f': masc_count/fem_count if fem_count > 0 else float('inf')
    })

gendered_df = pd.DataFrame(gendered_df_data)
gendered_df.to_csv("claude-gendered-words.csv", index=False)

# Print Action vs Appearance Analysis
print("\n\n3. ACTION VS APPEARANCE LANGUAGE")
print("-" * 80)

action_appearance_df_data = []
for gender in genders:
    action_count = action_appearance_results[gender]['action']
    appearance_count = action_appearance_results[gender]['appearance']
    total = action_count + appearance_count

    action_pct = (action_count / total * 100) if total > 0 else 0
    appearance_pct = (appearance_count / total * 100) if total > 0 else 0

    print(f"\n{gender.upper()}:")
    print(f"  Action words:     {action_count:4d} ({action_pct:5.2f}%)")
    print(f"  Appearance words: {appearance_count:4d} ({appearance_pct:5.2f}%)")
    print(f"  Ratio (A/App):    {action_count/appearance_count if appearance_count > 0 else 'inf'}")

    action_appearance_df_data.append({
        'gender': gender,
        'action_count': action_count,
        'appearance_count': appearance_count,
        'action_pct': action_pct,
        'appearance_pct': appearance_pct,
        'ratio_action_appearance': action_count/appearance_count if appearance_count > 0 else float('inf')
    })

action_appearance_df = pd.DataFrame(action_appearance_df_data)
action_appearance_df.to_csv("claude-action-appearance.csv", index=False)

# Top words by gender
print("\n\n4. TOP 20 WORDS BY GENDER")
print("-" * 80)

top_words_data = []
for gender in genders:
    word_counter = Counter(all_words_by_gender[gender])
    print(f"\n{gender.upper()}:")
    for word, count in word_counter.most_common(20):
        print(f"  {word:20s}: {count:4d}")
        top_words_data.append({
            'gender': gender,
            'word': word,
            'count': count
        })

top_words_df = pd.DataFrame(top_words_data)
top_words_df.to_csv("claude-top-words.csv", index=False)

# Statistical testing for gendered words
print("\n\n5. STATISTICAL SIGNIFICANCE TESTS")
print("-" * 80)

# Chi-square test for masculine vs feminine words across genders
observed = []
for gender in ['male', 'female']:
    masc = gendered_word_results[gender]['masculine']
    fem = gendered_word_results[gender]['feminine']
    observed.append([masc, fem])

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-square test (Male vs Female for gendered words):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

# Chi-square test for action vs appearance words
observed2 = []
for gender in ['male', 'female']:
    action = action_appearance_results[gender]['action']
    appearance = action_appearance_results[gender]['appearance']
    observed2.append([action, appearance])

chi2_2, p_value_2, dof_2, expected_2 = stats.chi2_contingency(observed2)
print(f"\nChi-square test (Male vs Female for action/appearance words):")
print(f"  Chi-square statistic: {chi2_2:.4f}")
print(f"  p-value: {p_value_2:.4f}")
print(f"  Significant: {'YES' if p_value_2 < 0.05 else 'NO'}")

print("\n" + "=" * 80)
print("Analysis complete! Results saved to CSV files.")
print("=" * 80)
