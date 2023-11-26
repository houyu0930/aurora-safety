import spacy
import re
import random
import datetime
from datasets import load_dataset
from faker_manager import augment_anonymize, faker_list

from constant import *


def swap_gender(text, main_gender, threshold):
    # TODO: make the mappings more inclusive (its, it, they, them)
    if main_gender == "male":
        gender_swap_dict = female_to_male_gender_swap
    else:
        gender_swap_dict = male_to_female_gender_swap

    def replace_gender_words(match):
        word = match.group(0)
        if random.random() < threshold:
            return word
        replacement = gender_swap_dict.get(word.lower(), word)
        return replacement if not word[0].isupper() else replacement.capitalize()

    pattern = r'\b(' + '|'.join(re.escape(key) for key in gender_swap_dict.keys()) + r')\b'
    clean_text = re.sub(pattern, replace_gender_words, text, flags=re.I)

    return clean_text


def swap_token(text, option_list):
    text_tokens = text.split()
    for idx, token in enumerate(text_tokens):
        if token.lower() in set(option_list):
            new_token = random.choice(option_list)
            if token[0].isupper():
                new_token = new_token.capitalize()
            text_tokens[idx] = new_token
    return ' '.join(text_tokens)


def augment_fake_name(text, entities, lang='en'):
    # TODO: 1) Swap first name only, 2) consider gender
    augment_anonymize_results = augment_anonymize(text, lang, entities)
    return augment_anonymize_results[0], (augment_anonymize_results[1], augment_anonymize_results[2])


def process_wiki_text_chunk(text, main_gender, lang):
    doc = nlp(text)

    # 1. Augment with fake first name
    entities = [(entity.text, entity.start_char, entity.end_char, entity.label_)
                for entity in doc.ents if entity.label_ in ['PERSON']]
    fake_name_meta = None
    if len(entities) > 0:
        text, fake_name_meta = augment_fake_name(text, entities, lang)

    # 2. Swap gender: as both male or female could exist,
    text = swap_gender(text, main_gender, threshold=0.2)

    # 3. Randomly choice race
    text = swap_token(text, race_list)

    # 4. Randomly choice religion
    text = swap_token(text, religion_list)

    return text, fake_name_meta


def augment_wikidata(record):
    wiki_text = record['text']

    # Split long wiki text into paragraphs
    wiki_text_chunks = wiki_text.split('\n\n')
    selected_lang = random.choice(lang_list)

    augmented_wiki_text_chunks = []
    fake_name_meta_list = []
    for chunk in wiki_text_chunks:
        # Randomly decide the main gender ot this chunk
        main_gender = random.choice(['male', 'female'])
        updated_chunk, fake_name_meta = process_wiki_text_chunk(chunk, main_gender, selected_lang)
        augmented_wiki_text_chunks.append(updated_chunk)
        fake_name_meta_list.append(fake_name_meta)
    text_after_augment = '\n\n'.join(augmented_wiki_text_chunks)

    return {
        'id': record['id'],
        'text_after_augment': text_after_augment,
        'fake_name_meta': fake_name_meta_list,
        'lang': selected_lang,
        'text_before_augment': record['text']
    }


def main():
    # https://huggingface.co/datasets/wikipedia
    start_time = datetime.datetime.now()
    dataset = load_dataset('wikipedia', '20220301.en')['train']

    augmented_wikidata = dataset.map(augment_wikidata)
    augmented_wikidata.save_to_disk(f'aurora-safety-augmented-wikidata')

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == '__main__':
    random.seed(42)
    nlp = spacy.load("en_core_web_lg")
    lang_list = [faker_lang.split("_")[0] for faker_lang in faker_list]
    main()
