import random
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
import json
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("stopwords")
summarizer = pipeline("summarization", model="t5-base")

STOPWORDS = set(stopwords.words("english"))


def extractTextFromJson(path, output):
    if not os.path.exists(output):
        with open(path, 'r') as file:
            data = json.load(file)

        rows = []
        for entry in data:
            if entry['ANSWERABLE'] == 'Y':
                question = str(entry['QUESTION_TITLE'] + ' ' + entry['QUESTION_TEXT'])
                answer = str(entry['ANSWER'])
                rows.append({'Question': question, 'Answer': answer})

        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)


def extractTextFromPdf(path):
    with fitz.open(path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()

    sectionPattern = re.compile(r'\d+\.\d+.*')
    sections = sectionPattern.split(text)
    headers = sectionPattern.findall(text)

    combinedSections = [
        "{} {}".format(header, text.replace('\n', ' '))
        for header, text in zip(headers, sections[1:])
    ]
    return combinedSections


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
    userManualDataset = "data/userManualDataset.csv"
    userManualPath = "data/SIMetrixUsersManual.pdf"

    if not os.path.exists(userManualDataset):
        sections = extractTextFromPdf(userManualPath)
        createDataset(sections, userManualDataset)

    df = pd.read_csv(userManualDataset)
    return df['text'].tolist()


def generateQuestion(header):
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

    if len(section.split()) < maxInLen:
        return section

    return ' '.join(sent_tokenize(section)[:3])


def createQaPairs(headers, sections):
    qa_pairs = []

    for header in headers:
        best_section = extractBestMatch(header, sections)

        question = generateQuestion(header)
        answer = extractAnswer(best_section)

        qa_pairs.append((question, answer))

    return qa_pairs


def generateQaPairs():
    sections = getSections()

    headersCsv = pd.read_csv("data/keywords.csv")
    headers = headersCsv['text'].tolist()

    pairsFile = "data/qaPairs.csv"
    manuallyCreatedFile = "data/manuallyCreated.csv"

    if not os.path.exists(pairsFile):
        headers = [header for header in headers if pd.notna(header) and header.strip()]

        sections = [section for section in sections if pd.notna(section) and section.strip()]

        qaPairs = createQaPairs(headers, sections)

        df = pd.DataFrame(qaPairs, columns=["Question", "Answer"])

        if os.path.exists(manuallyCreatedFile):
            manually_created_df = pd.read_csv(manuallyCreatedFile)
            df = pd.concat([df, manually_created_df], ignore_index=True)
        else:
            print(f"Manually created file {manuallyCreatedFile} does not exist.")

        df.to_csv(pairsFile, index=False)


def generateSynonymPermutations(text, num_permutations=5):
    words = text.split()
    permutations = []

    for _ in range(num_permutations):
        newWords = []
        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonymList = synonyms[0].lemmas()
                if synonymList:
                    newWord = synonymList[random.randint(0, len(synonymList)-1)].name()
                    newWords.append(newWord.replace('_', ' '))
                else:
                    newWords.append(word)
            else:
                newWords.append(word)
        permutations.append(" ".join(newWords))

    return permutations


def augmentQaPairs():
    pairsFile = "data/qaPairsAug.csv"
    originalFile = "data/qaPairs.csv"

    if not os.path.exists(originalFile):
        print(f"Original file {originalFile} does not exist.")
        return

    df = pd.read_csv(originalFile)

    augRows = []
    for index, row in df.iterrows():
        origQ = row['Question']
        permutations = generateSynonymPermutations(origQ, num_permutations=5)
        for newQ in permutations:
            newRow = row.copy()
            newRow['Question'] = newQ
            augRows.append(newRow)

    augDf = pd.DataFrame(augRows)

    resultDf = pd.concat([df, augDf], ignore_index=True)

    resultDf.to_csv(pairsFile, index=False)


