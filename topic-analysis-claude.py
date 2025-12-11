import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict, Counter
import spacy

# Load spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Define semantic field categories
STEM_WORDS = {
    'science', 'technology', 'engineering', 'math', 'mathematics', 'physics',
    'chemistry', 'biology', 'computer', 'coding', 'programming', 'robot',
    'robotics', 'electronics', 'circuit', 'experiment', 'laboratory', 'research',
    'calculator', 'telescope', 'microscope', 'technical', 'build', 'construct',
    'lego', 'blocks', 'building', 'mechanic', 'mechanical', 'engineer', 'astronaut'
}

ARTS_HUMANITIES_WORDS = {
    'art', 'music', 'dance', 'painting', 'drawing', 'coloring', 'creative',
    'singing', 'instrument', 'piano', 'guitar', 'crafts', 'literature', 'reading',
    'writing', 'poetry', 'story', 'theater', 'drama', 'acting', 'performance',
    'ballet', 'sculpture', 'design', 'artistic', 'sketch', 'canvas', 'culture',
    'history', 'language', 'book', 'novel', 'creative', 'imagination'
}

PHYSICAL_ACTIVITIES = {
    'sports', 'soccer', 'basketball', 'football', 'baseball', 'tennis', 'swimming',
    'running', 'athletic', 'exercise', 'gym', 'cycling', 'skateboard', 'outdoor',
    'hiking', 'climbing', 'jumping', 'playing', 'active', 'physical', 'fitness',
    'martial', 'karate', 'wrestling', 'track', 'field', 'game', 'ball', 'ride'
}

SOCIAL_EMOTIONAL_WORDS = {
    'friend', 'friendship', 'social', 'communication', 'talk', 'share', 'caring',
    'helping', 'helper', 'kindness', 'empathy', 'emotional', 'feelings', 'caring',
    'nurture', 'nurturing', 'teacher', 'teaching', 'learning', 'cooperation',
    'teamwork', 'community', 'family', 'together', 'group', 'relationship'
}

def preprocess_text(text):
    """Clean and lemmatize text."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(tokens)

def classify_semantic_field(text):
    """Classify text into semantic fields."""
    doc = nlp(text.lower())
    lemmas = {token.lemma_ for token in doc if token.is_alpha}

    fields = {
        'STEM': len(lemmas & STEM_WORDS),
        'Arts_Humanities': len(lemmas & ARTS_HUMANITIES_WORDS),
        'Physical': len(lemmas & PHYSICAL_ACTIVITIES),
        'Social_Emotional': len(lemmas & SOCIAL_EMOTIONAL_WORDS)
    }
    return fields

# Load data
INPUT_JSON = "claude3results-sanitized.json"
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=" * 80)
print("TOPICAL DISTRIBUTION ANALYSIS - CLAUDE RESULTS")
print("=" * 80)

# Preprocess all responses
df['processed_response'] = df['prompt-response'].apply(preprocess_text)

# Categories and groups
categories = ["hobbies", "toys", "careers", "academics"]
genders = ["male", "female", "child"]

# ============================================================================
# 1. TF-IDF KEYWORD EXTRACTION BY GENDER
# ============================================================================
print("\n1. TF-IDF KEYWORD EXTRACTION BY GENDER")
print("-" * 80)

tfidf_results = {}
for gender in genders:
    gender_texts = df[df['gender'] == gender]['processed_response'].tolist()

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(gender_texts)

    # Get feature names and scores
    feature_names = tfidf.get_feature_names_out()
    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()

    # Sort by importance
    top_indices = mean_tfidf.argsort()[-20:][::-1]
    top_keywords = [(feature_names[i], mean_tfidf[i]) for i in top_indices]

    tfidf_results[gender] = top_keywords

    print(f"\n{gender.upper()} - Top 20 Keywords:")
    for keyword, score in top_keywords:
        print(f"  {keyword:30s}: {score:.4f}")

# Save TF-IDF results
tfidf_df_data = []
for gender, keywords in tfidf_results.items():
    for keyword, score in keywords:
        tfidf_df_data.append({
            'gender': gender,
            'keyword': keyword,
            'tfidf_score': score
        })
tfidf_df = pd.DataFrame(tfidf_df_data)
tfidf_df.to_csv("claude-tfidf-keywords.csv", index=False)

# ============================================================================
# 2. TOPIC MODELING (LDA)
# ============================================================================
print("\n\n2. TOPIC MODELING (LDA) - 8 TOPICS")
print("-" * 80)

# Prepare data for LDA
all_texts = df['processed_response'].tolist()

# Count vectorization for LDA
count_vectorizer = CountVectorizer(max_features=100, min_df=3, max_df=0.8)
count_matrix = count_vectorizer.fit_transform(all_texts)

# LDA model
n_topics = 8
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=50)
lda.fit(count_matrix)

# Get topic-word distributions
feature_names = count_vectorizer.get_feature_names_out()

print("\nTopics (top 10 words per topic):")
topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")
    topic_words.append(top_words)

# Get document-topic distributions
doc_topic_dist = lda.transform(count_matrix)
df['topic_distribution'] = list(doc_topic_dist)
df['dominant_topic'] = doc_topic_dist.argmax(axis=1)

# Topic distribution by gender
print("\n\nTopic Distribution by Gender:")
print("-" * 80)

topic_gender_data = []
for gender in genders:
    gender_df = df[df['gender'] == gender]
    topic_counts = Counter(gender_df['dominant_topic'])
    total = len(gender_df)

    print(f"\n{gender.upper()}:")
    for topic_id in range(n_topics):
        count = topic_counts[topic_id]
        pct = (count / total) * 100
        print(f"  Topic {topic_id + 1}: {count:4d} ({pct:5.2f}%)")
        topic_gender_data.append({
            'gender': gender,
            'topic': topic_id + 1,
            'count': count,
            'percentage': pct
        })

topic_gender_df = pd.DataFrame(topic_gender_data)
topic_gender_df.to_csv("claude-topic-gender-distribution.csv", index=False)

# ============================================================================
# 3. SEMANTIC FIELD CLASSIFICATION
# ============================================================================
print("\n\n3. SEMANTIC FIELD CLASSIFICATION")
print("-" * 80)

# Classify each response
semantic_field_counts = defaultdict(lambda: defaultdict(int))

for idx, row in df.iterrows():
    gender = row['gender']
    fields = classify_semantic_field(row['prompt-response'])

    for field, count in fields.items():
        semantic_field_counts[gender][field] += count

# Calculate totals and percentages
semantic_field_data = []
print("\nSemantic Field Distribution by Gender:")
for gender in genders:
    total = sum(semantic_field_counts[gender].values())
    print(f"\n{gender.upper()} (Total mentions: {total}):")

    for field in ['STEM', 'Arts_Humanities', 'Physical', 'Social_Emotional']:
        count = semantic_field_counts[gender][field]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {field:20s}: {count:4d} ({pct:5.2f}%)")
        semantic_field_data.append({
            'gender': gender,
            'semantic_field': field,
            'count': count,
            'percentage': pct
        })

semantic_field_df = pd.DataFrame(semantic_field_data)
semantic_field_df.to_csv("claude-semantic-fields.csv", index=False)

# ============================================================================
# 4. CATEGORY-SPECIFIC ANALYSIS
# ============================================================================
print("\n\n4. CATEGORY-SPECIFIC SEMANTIC FIELD ANALYSIS")
print("-" * 80)

category_semantic_data = []
for category in categories:
    print(f"\n{category.upper()}:")
    for gender in genders:
        cat_gender_df = df[(df['category'] == category) & (df['gender'] == gender)]

        field_counts = defaultdict(int)
        for _, row in cat_gender_df.iterrows():
            fields = classify_semantic_field(row['prompt-response'])
            for field, count in fields.items():
                field_counts[field] += count

        total = sum(field_counts.values())
        print(f"  {gender:10s}:", end='')
        for field in ['STEM', 'Arts_Humanities', 'Physical', 'Social_Emotional']:
            count = field_counts[field]
            pct = (count / total * 100) if total > 0 else 0
            print(f" {field}={pct:4.1f}%", end='')
            category_semantic_data.append({
                'category': category,
                'gender': gender,
                'semantic_field': field,
                'count': count,
                'percentage': pct
            })
        print()

category_semantic_df = pd.DataFrame(category_semantic_data)
category_semantic_df.to_csv("claude-category-semantic-fields.csv", index=False)

# ============================================================================
# 5. WORD CO-OCCURRENCE ANALYSIS
# ============================================================================
print("\n\n5. WORD CO-OCCURRENCE ANALYSIS")
print("-" * 80)

def get_word_pairs(text):
    """Extract word pairs from text."""
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct and token.is_alpha]
    pairs = []
    for i in range(len(lemmas) - 1):
        pairs.append((lemmas[i], lemmas[i+1]))
    return pairs

cooccurrence_by_gender = defaultdict(Counter)

for idx, row in df.iterrows():
    gender = row['gender']
    pairs = get_word_pairs(row['prompt-response'])
    cooccurrence_by_gender[gender].update(pairs)

print("\nTop 10 Word Pairs by Gender:")
cooccurrence_data = []
for gender in genders:
    print(f"\n{gender.upper()}:")
    for pair, count in cooccurrence_by_gender[gender].most_common(10):
        print(f"  {pair[0]:15s} + {pair[1]:15s}: {count:3d}")
        cooccurrence_data.append({
            'gender': gender,
            'word1': pair[0],
            'word2': pair[1],
            'count': count
        })

cooccurrence_df = pd.DataFrame(cooccurrence_data)
cooccurrence_df.to_csv("claude-word-cooccurrence.csv", index=False)

print("\n" + "=" * 80)
print("Topic analysis complete! Results saved to CSV files.")
print("=" * 80)
