import sys
import codecs
import json
import re
from tqdm import tqdm
import numpy as np
import random
from nltk import word_tokenize
from ordered_set import OrderedSet

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def generating_questions():
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    all_summary_features = json_load('data/xsum_linguistic_features.json')

    generated_questions = {}

    for key in tqdm(all_summary_features):
        e = all_summary_features[key]
        generated_questions[key] = {'article': e['article'], 'summary': e['sentence'], 'qas': {}}

        # we only focus on the main verb
        if 'ROOT' not in e['dp']:
            continue
        # we only generate questions based on the main verb of the summary
        main_verb_ind = e['dp'].index('ROOT')
        main_verb = e['tokens'][main_verb_ind]

        # the factor controling the probability of using heuristically constructed wh-word
        perb_factor = 9
        for frame in e['srl_frames']:
            # all argument tags
            tags = frame['tags']
            tags = OrderedSet(t[2:] for t in tags if t != 'O')

            # extract all argument strings
            res = re.findall(r'\[.*?\]', frame['description'])
            res = [r[1:-1] for r in res if ':' in r]
            res_str = [r.split(':')[1].strip() for r in res]

            arg_dict = {arg: string for arg, string in zip(tags, res_str)}

            if frame['verb'] == main_verb:
                summary_sent = e['sentence'].strip()[:-1]

                # heuristics that can be re-used for other arguments (such as ARGM-TMP) with specific adjustments
                if 'says' in summary_sent and summary_sent.index('says') == (len(summary_sent) - len('says')):
                    r_summary_sent = [summary_sent[i] for i in range(len(summary_sent) - 1, -1, -1)]
                    who_says = summary_sent[len(summary_sent) - r_summary_sent.index(','):].strip()
                    who_says = who_says[0].upper() + who_says[1:]

                    summary_sent = summary_sent.replace(who_says, '')
                    summary_sent = summary_sent[0].lower() + summary_sent[1:]
                    summary_sent = who_says + ' ' + summary_sent
                if 'ARG0' in arg_dict and 'ARG1' in arg_dict:
                    if arg_dict['ARG0'] in summary_sent:
                        _summary = summary_sent.replace(arg_dict['ARG0'], '').strip()
                        if not _summary.split()[0].isupper():
                            _summary = _summary[0].lower() + _summary[1:]

                        wh_word = 'What '
                        ans_word = arg_dict['ARG0']
                        if tags.index('ARG0') < tags.index('ARG1'):
                            # NER wh-word heuristics
                            for _ner in e['ner']:
                                if arg_dict['ARG0'].endswith(_ner[0]):
                                    wh_word = 'Which ' + arg_dict['ARG0'].replace(_ner[0], '')
                                    ans_word = _ner[0]
                                    break
                                if _ner[0] in arg_dict['ARG0'] and _ner[1] == 'PERSON':
                                    wh_word = 'Who '
                                    break

                        # decompose verb and aux word
                        lem_verb = lemmatizer.lemmatize(main_verb)
                        if main_verb == lem_verb:
                            if tags.index('ARG0') > tags.index('ARG1'):
                                v_ind = e['tokens'].index(main_verb)
                                aux_word = e['tokens'][v_ind - 1]

                                # if aux word not in arg, it should be a valid aux word
                                if aux_word not in arg_dict['ARG1']:
                                    wh_word = wh_word + aux_word + ' '
                                    _summary = _summary.replace(aux_word, '').strip()
                                else:
                                    wh_word = wh_word + 'do '
                            if random.randint(0, 10) < perb_factor:
                                _question = wh_word + _summary + ' ?'
                            else:
                                _question = 'What ' + _summary + ' ?'

                        else:
                            if tags.index('ARG0') > tags.index('ARG1'):
                                v_ind = e['tokens'].index(main_verb)
                                aux_word = e['tokens'][v_ind - 1]

                                if aux_word not in arg_dict['ARG1']:
                                    wh_word = wh_word + aux_word + ' '
                                    _summary = _summary.replace(aux_word, '').strip()
                                else:
                                    if main_verb.endswith('ed'):
                                        wh_word = wh_word + 'did '
                                    if main_verb.endswith('s'):
                                        wh_word = wh_word + 'does '
                            if random.randint(0, 10) < perb_factor:
                                _question = wh_word + _summary + ' ?'
                            else:
                                _question = 'What ' + _summary + ' ?'
                        _question = _question.replace('  ', ' ')
                        _question = _question.replace(',', '')
                        generated_questions[key]['qas']['ARG0'] = {'question': _question,
                                                                   'answer': ans_word}

                    if arg_dict['ARG1'] in summary_sent:
                        _summary = summary_sent.replace(arg_dict['ARG1'], '').strip()
                        if not _summary.split()[0].isupper():
                            _summary = _summary[0].lower() + _summary[1:]

                        wh_word = 'What '
                        ans_word = arg_dict['ARG1']

                        if tags.index('ARG0') > tags.index('ARG1'):
                            for _ner in e['ner']:
                                if arg_dict['ARG1'].endswith(_ner[0]):
                                    wh_word = 'Which ' + arg_dict['ARG1'].replace(_ner[0], '')
                                    ans_word = _ner[0]
                                    break
                                if _ner[0] in arg_dict['ARG1'] and _ner[1] == 'PERSON':
                                    wh_word = 'Who '
                                    break

                        lem_verb = lemmatizer.lemmatize(main_verb)
                        if main_verb == lem_verb:
                            if tags.index('ARG1') > tags.index('ARG0'):
                                v_ind = e['tokens'].index(main_verb)
                                aux_word = e['tokens'][v_ind - 1]

                                if aux_word not in arg_dict['ARG1']:
                                    wh_word = wh_word + aux_word + ' '
                                    _summary = _summary.replace(aux_word, '').strip()
                                else:
                                    wh_word = wh_word + 'do '
                            if random.randint(0, 10) < perb_factor:
                                _question = wh_word + _summary + ' ?'
                            else:
                                _question = 'What ' + _summary + ' ?'

                        else:
                            if tags.index('ARG1') > tags.index('ARG0'):
                                v_ind = e['tokens'].index(main_verb)
                                aux_word = e['tokens'][v_ind - 1]

                                if aux_word not in arg_dict['ARG0']:
                                    wh_word = wh_word + aux_word + ' '
                                    _summary = _summary.replace(aux_word, '').strip()
                                else:
                                    if main_verb.endswith('ed'):
                                        wh_word = wh_word + 'did '
                                    if main_verb.endswith('s'):
                                        wh_word = wh_word + 'does '
                            if random.randint(0, 10) < perb_factor:
                                _question = wh_word + _summary + ' ?'
                            else:
                                _question = 'What ' + _summary + ' ?'
                        _question = _question.replace('  ', ' ')
                        _question = _question.replace(',', '')
                        generated_questions[key]['qas']['ARG1'] = {'question': _question,
                                                                   'answer': ans_word}

                # basic version with more heuristics for simplicity
                if 'ARGM-CAU' in arg_dict:
                    _summary = summary_sent.replace(arg_dict['ARGM-CAU'], '').strip()
                    if not _summary.split()[0].isupper():
                        _summary = _summary[0].lower() + _summary[1:]
                    _question = 'Why ' + _summary + ' ?'

                    _question = _question.replace('  ', ' ')
                    _question = _question.replace(',', '')
                    generated_questions[key]['qas']['ARGM-CAU'] = {'question': _question,
                                                                   'answer': arg_dict['ARGM-CAU']}

                # basic version with more heuristics for simplicity
                if 'ARGM-MNR' in arg_dict:
                    _summary = summary_sent.replace(arg_dict['ARGM-MNR'], '').strip()
                    if not _summary.split()[0].isupper():
                        _summary = _summary[0].lower() + _summary[1:]
                    _question = 'How ' + _summary + ' ?'

                    _question = _question.replace('  ', ' ')
                    _question = _question.replace(',', '')
                    generated_questions[key]['qas']['ARGM-MNR'] = {'question': _question,
                                                                   'answer': arg_dict['ARGM-MNR']}

                # basic version with more heuristics for simplicity
                if 'ARGM-LOC' in arg_dict:
                    _summary = summary_sent.replace(arg_dict['ARGM-LOC'], '').strip()
                    if not _summary.split()[0].isupper():
                        _summary = _summary[0].lower() + _summary[1:]
                    _question = 'Where ' + _summary + ' ?'

                    _question = _question.replace('  ', ' ')
                    _question = _question.replace(',', '')
                    generated_questions[key]['qas']['ARGM-LOC'] = {'question': _question,
                                                                   'answer': arg_dict['ARGM-LOC']}

                # basic version with more heuristics for simplicity
                if 'ARGM-TMP' in arg_dict:
                    _summary = summary_sent.replace(arg_dict['ARGM-TMP'], '').strip()
                    if not _summary.split()[0].isupper():
                        _summary = _summary[0].lower() + _summary[1:]
                    _question = 'When ' + _summary + ' ?'

                    _question = _question.replace('  ', ' ')
                    _question = _question.replace(',', '')
                    generated_questions[key]['qas']['ARGM-TMP'] = {'question': _question,
                                                                   'answer': arg_dict['ARGM-TMP']}

                # basic version with more heuristics for simplicity
                if 'ARGM-PRP' in arg_dict:
                    _summary = summary_sent.replace(arg_dict['ARGM-PRP'], '').strip()
                    if not _summary.split()[0].isupper():
                        _summary = _summary[0].lower() + _summary[1:]
                    _question = 'For what purpose ' + _summary + ' ?'

                    _question = _question.replace('  ', ' ')
                    _question = _question.replace(',', '')
                    generated_questions[key]['qas']['ARGM-PRP'] = {'question': _question,
                                                                   'answer': arg_dict['ARGM-PRP']}

                break
    return generated_questions


if __name__ == '__main__':
    generated_questions = generating_questions()
