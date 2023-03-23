import sys
import codecs
import json
from tqdm import tqdm
from nltk import word_tokenize


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def get_context(data):
    with open(data, 'r') as f:
        all_lines = f.readlines()
    context = {}
    for sample in tqdm(all_lines):
        j = json.loads(sample)
        context[j['id']] = [j['document'], j['summary']]
    return context


def get_dependency(sent, dependency_parser):
    if len(sent.strip()) == 0:
        return None
    try:
        sent = dependency_parser.predict(sentence=sent)
    except:
        import ipdb; ipdb.set_trace()
    words, pos, heads, dependencies = sent['words'], sent['pos'], sent['predicted_heads'], sent['predicted_dependencies']
    result = [{'word':w, 'pos':p, 'head':h - 1, 'dep':d} for w, p, h, d in zip(words, pos, heads, dependencies)]
    return result


def dependency_parse(raw, filename):
    dependency_parser = Predictor.from_path("dependency_parse/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    context = {
        key: [
            get_dependency(sent, dependency_parser) for sent in value
        ] for key, value in tqdm(raw.items(), desc='   - (Dependency Parsing: 1st) -   ')
    }
    json_dump(context, filename)


def get_coreference(doc, coref_reslt, pronouns, title):

    def get_crf(span, words):
        phrase = []
        for i in range(span[0], span[1] + 1):
            phrase += [words[i]]
        return (' '.join(phrase), span[0], span[1] - span[0] + 1)

    def get_best(crf):
        crf.sort(key=lambda x: x[2], reverse=True)
        if crf[0][2] == 1:
            crf.sort(key=lambda x: len(x[0]), reverse=True)
        for w in crf:
            if w[0].lower() not in pronouns and w[0].lower() != '\t':
                return w[0]
        return None

    doc = coref_reslt.predict(document=doc)
    words = [w.strip(' ') for w in doc['document']]
    clusters = doc['clusters']

    for group in clusters:
        crf = [get_crf(span, words) for span in group]
        entity = get_best(crf)
        if entity in ['\t', None]:
            try:
                entity = coref_reslt.predict(document=title)
                entity = ' '.join(entity['document'])
            except:
                entity = ' '.join(word_tokenize(title))
        if entity not in ['\t', None]:
            for phrase in crf:
                if phrase[0].lower() in pronouns:
                    index = phrase[1]
                    words[index] = entity

    doc, sent = [], []
    for word in words:
        if word.strip(' ') == '\t':
            doc.append(sent)
            sent = []
        else:
            if word.count('\t'):
                print(word)
                word = word.strip('\t')
            sent.append(word)
    doc.append(sent)
    return doc


def coreference_resolution(raw, filename):
    pronouns = ['it', 'its', 'he', 'him', 'his', 'she', 'her', 'they', 'their', 'them']
    raw = {d[0]: '\t'.join(d[1]) for d in raw}
    coref_reslt = Predictor.from_path("coreference_resolution/coref-model-2018.02.05.tar.gz")
    context = {
        key: get_coreference(value, coref_reslt, pronouns, key) for key, value in tqdm(raw.items(), desc='  - (crf for evidence) ')
    }
    json_dump(context, filename)


def get_ner(doc, ner_tagger):
    try:
        doc = ner_tagger.predict(sentence=doc)
    except:
        return [[doc, 'O']]
    words, tags = doc['words'], doc['tags']
    return [[w, t] for w, t in zip(words, tags)]


def ner_tag(raw, filename):
    ner_tagger = Predictor.from_path("ner_tag/ner-model-2018.12.18.tar.gz")
    raw = [[d[0], d[0]] for d in raw]    #raw = [[d[0], '\t'.join(d[1])] for d in raw]
    context = {sample[0]: get_ner(sample[1], ner_tagger) for sample in tqdm(raw, desc='  - (ner for evidence) ')}
    json_dump(context, filename)


def sr_labeling(sent, sr_labeler):
    if len(sent.strip()) == 0:
        return None
    try:
        sent = sr_labeler.predict(sentence=sent)
    except:
        import ipdb; ipdb.set_trace()
    length, words, verbs = len(sent['words']), sent['words'], sent['verbs']
    return {'srl_frames': verbs}


def semantic_role_labeling(raw):
    sr_labeler = Predictor.from_path("pre_trained_models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    outputs = {sample: sr_labeling(raw[sample][1], sr_labeler)
               for sample in tqdm(raw, desc='   - (Semantic Role Labeling: 1st) -   ')}
    return outputs


def dp_and_ner_from_spacy(raw):
    import spacy
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    outputs = {}
    for key in tqdm(raw, desc='   - (Dependency Parsing and Named Entity Recognition: 1st) -   '):
        sent = raw[key][1]
        doc = nlp(sent)
        outputs[key] = {'dp': [t.dep_ for t in doc], 'tokens': [t.text for t in doc],
                        'ner': [[entity.text, entity.label_]for entity in doc.ents],
                        'sentence': sent, 'article': raw[key][0]}
    return outputs


def export_xsum_dataset_from_huggingface():
    from datasets import load_dataset
    dataset = load_dataset('xsum')
    # print(dataset)
    dataset['train'].to_json('data/xsum_train.json')
    dataset['validation'].to_json('data/xsum_dev.json')
    dataset['test'].to_json('data/xsum_test.json')


def find_all_arguments():
    all_features = json_load('data/xsum_linguistic_features.json')

    srl_args = {}
    for e in all_features:
        e = all_features[e]
        for frame in e['srl_frames']:
            tags = frame['tags']
            tags = set(t[2:] for t in tags if t != 'O')
            if 'R-ARG0' in tags:
                a = 1
            for t in tags:
                if t not in srl_args:
                    srl_args[t] = 0
                srl_args[t] += 1

    print(srl_args)


if __name__ == '__main__':
    export_xsum_dataset_from_huggingface()
    from allennlp.predictors.predictor import Predictor
    train_data_file, valid_data_file = 'data/xsum_train.json', 'data/xsum_dev.json'
    context = get_context(valid_data_file)
    print('number of context:', len(context))

    # dependency_parse(context)
    # coreference_resolution(context)
    # ner_tag(context)
    srl_outptus = semantic_role_labeling(context)
    dp_ner_outputs = dp_and_ner_from_spacy(context)
    outputs = {}
    for key in srl_outptus:
        srl_outptus[key].update(dp_ner_outputs[key])
        outputs[key] = srl_outptus[key]
    # print(outputs)
    json_dump(outputs, 'data/xsum_linguistic_features.json')
    pass

