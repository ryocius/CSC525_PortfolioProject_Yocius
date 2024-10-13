import nltk
import os
import re
import fitz
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download("punkt")
nltk.download("stopwords")
summarizer = pipeline("summarization", model="t5-base")

STOPWORDS = set(stopwords.words("english"))

def extractTextFromPdf(path):
    with fitz.open(path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()

    section_pattern = re.compile(r'\d+\.\d+.*')
    sections = section_pattern.split(text)
    headers = section_pattern.findall(text)

    combined_sections = [
        "{} {}".format(header, text.replace('\n', ' '))
        for header, text in zip(headers, sections[1:])
    ]
    return combined_sections

def preprocessText(text):
    text = re.sub("SIMetrix/SIMPLIS User’s Manual", "", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [word for word in word_tokenize(text) if word not in STOPWORDS]
    return ' '.join(words)

def createDataset(sections, filename):
    filtered_sections = [
        section for section in sections
        if len(re.sub(r'[^a-zA-Z0-9\s]', '', section)) > 15
    ]
    data = {'text': [preprocessText(section) for section in filtered_sections]}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def getSections():
    userManualDataset = "userManualDataset.csv"
    userManualPath = "SIMetrixUsersManual.pdf"

    if not os.path.exists(userManualDataset):
        sections = extractTextFromPdf(userManualPath)
        createDataset(sections, userManualDataset)

    df = pd.read_csv(userManualDataset)
    return df['text'].tolist()

def generate_question(header):
    header = header.lower()

    if "installation" in header:
        return f"How do you install the {header}?"

    if "removing" in header or "deleting" in header:
        return f"How can you remove or delete {header}?"

    if "overview" in header:
        return f"What is the overview of {header}?"

    if "editing" in header or "modify" in header:
        return f"How can {header} be edited?"

    if "importing" in header:
        return f"How do you import {header}?"


    return f"What does the '{header}' section cover?"

def extractBestMatch(header, sections):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([header] + sections)
    cosSimilarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    bestMatch = cosSimilarity.argmax()
    return sections[bestMatch]


def extractAnswer(section):
    maxInLen = 1024
    maxOutLen = 150

    if len(section.split()) < maxInLen:
        summary = summarizer(section, max_length=maxOutLen, min_length=5, do_sample=False)
        print(summary)
        return summary[0]['summary_text']

    return ' '.join(sent_tokenize(section)[:3])

def createQaPairs(headers, sections):
    qa_pairs = []

    for header in headers:
        best_section = extractBestMatch(header, sections)

        question = generate_question(header)
        answer = extractAnswer(best_section)

        qa_pairs.append((question, answer))

    return qa_pairs


def generateQaPairs():
    sections = getSections()

    headersCsv = pd.read_csv("keywords.csv")
    headers = headersCsv['text'].tolist()

    pairsFile = "qaPairs.csv"
    if not os.path.exists(pairsFile):
        headers = [header for header in headers if pd.notna(header) and header.strip()]

        sections = [section for section in sections if pd.notna(section) and section.strip()]

        qaPairs = createQaPairs(headers, sections)

        df = pd.DataFrame(qaPairs, columns=["Question", "Answer"])
        df.to_csv(pairsFile, index=False)


generateQaPairs()