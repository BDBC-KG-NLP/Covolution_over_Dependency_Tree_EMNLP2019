import os
import re
import sys
import json
import corenlp
import argparse
import xml.etree.ElementTree as ET

pos_dict = {
    'NN'    : 'NN',
    'DT'    : 'DT',
    'JJ'    : 'JJ',
    'IN'    : 'IN',
    'RB'    : 'RB',
    '.'     : '.',
    'PRP'   : 'PRP',
    'CC'    : 'CC',
    'VBD'   : 'VB',
    ','     : ',',
    'NNS'   : 'NN',
    'VB'    : 'VB',
    'VBZ'   : 'VB',
    'VBP'   : 'VB',
    'NNP'   : 'NN',
    'VBN'   : 'VB',
    'PRP$'  : 'PRP$',
    'TO'    : 'TO',
    'VBG'   : 'VB',
    'MD'    : 'MD',
    'CD'    : 'CD',
    'HYPH'  : 'HYPH',
    '-LRB-' : '-LRB-',
    '-RRB-' : '-RRB-',
    'WDT'   : 'WDT',
    'JJS'   : 'JJR',
    'RP'    : 'RP',
    'JJR'   : 'JJR',
    'WRB'   : 'WRB',
    'WP'    : 'WP',
    'POS'   : 'POS',
    '$'     : 'SYM',
    ':'     : ':',
    'EX'    : 'EX',
    'UH'    : 'UH',
    'PDT'   : 'PDT',
    'RBR'   : 'RBR',
    'RBS'   : 'RBR',
    'FW'    : 'FW',
    'NNPS'  : 'NN',
    'SYM'   : 'SYM',
    '\'\''  : 'SYM',
    '``'    : 'SYM',
    'NFP'   : 'NFP',
    'AFX'   : 'STM',
    'LS'    : 'SYM',
    'ADD'   : 'SYM',
    'GW'    : 'SYM',
    'WDT'   : 'WDT',
    'WP$'   : 'WP',
    '#'     : 'SYM'
}

dep_dict = {
    'punct'        : 'punct',
    'det'          : 'det',
    'case'         : 'case',
    'nsubj'        : 'subj',
    'advmod'       : 'advmod',
    'amod'         : 'amod',
    'cc'           : 'cc',
    'cop'          : 'cop',
    'conj'         : 'conj',
    'compound'     : 'compound',
    'obl'          : 'obl',
    'obj'          : 'obj',
    'mark'         : 'mark',
    'nmod'         : 'nmod',
    'aux'          : 'aux',
    'dep'          : 'dep',
    'nmod:poss'    : 'nmod:poss',
    'advcl'        : 'advcl',
    'nummod'       : 'nummod',
    'aux:pass'     : 'aux:pass',
    'parataxis'    : 'parataxis',
    'xcomp'        : 'xcomp',
    'ccomp'        : 'ccomp',
    'nsubj:pass'   : 'subj:pass',
    'appos'        : 'appos',
    'compound:prt' : 'compound',
    'acl:relcl'    : 'acl',
    'acl'          : 'acl',
    'fixed'        : 'fixed',
    'obl:npmod'    : 'obl',
    'obl:tmod'     : 'obl',
    'expl'         : 'expl',
    'iobj'         : 'obj',
    'det:predet'   : 'det',
    'discourse'    : 'dep',
    'cc:preconj'   : 'cc',
    'csubj'        : 'subj',
    'orphan'       : 'dep',
    'csubj:pass'   : 'subj:pass',
    'goeswith'     : 'dep'
}

def parseXML(data_path):
    tree = ET.ElementTree(file=data_path)
    objs = list()
    for sentence in tree.getroot():
        obj = dict()
        for item in sentence:
            if item.tag == 'text':
                obj['raw_text'] = item.text
            elif item.tag == 'aspectTerms':
                obj['aspects'] = list()
                for aspectTerm in item:
                    if aspectTerm.attrib['polarity'] != 'conflict':
                        obj['aspects'].append(aspectTerm.attrib)
        if 'aspects' in obj and len(obj['aspects']):
            objs.append(obj)
    return objs

def parseSentences(objs, client):
    succeed_objs = list()
    obj_num = len(objs)
    for k, obj in enumerate(objs):
        try:
            new_obj = {'token': list(), 'pos': list(), 'head': list(), 'deprel': list(), 'aspects': list()}
            raw_text = re.sub(r'[^\x00-\x7f]', ' ', obj['raw_text'])
            empty_num = 0
            while raw_text[0] == ' ':
                raw_text = raw_text[1:]
                empty_num += 1
            ann = client.annotate(raw_text)
            ''' Token and POS '''
            char_index = list()
            for token in ann.sentence[0].token:
                new_obj['token'].append(str(token.originalText))
                new_obj['pos'].append(pos_dict[str(token.pos)])
                char_index.append(int(str(token.beginChar))+empty_num)
            ''' Dependency '''
            dependency_parse = ann.sentence[0].basicDependencies
            connect_nodes = [list() for i in range(len(new_obj['token']))]
            deps = list()
            for edge in dependency_parse.edge:
                deps.append((int(str(edge.target)), int(str(edge.source)), dep_dict[str(edge.dep)]))
                connect_nodes[int(str(edge.target))-1].append(int(str(edge.source))-1)
                connect_nodes[int(str(edge.source))-1].append(int(str(edge.target))-1)
            deps.append((int(str(dependency_parse.root[0])), 0, "ROOT")) # 0 for root
            deps.sort()
            for _, head, deprel in deps:
                new_obj['head'].append(head)
                new_obj['deprel'].append(deprel)
            assert len(new_obj['token']) == len(new_obj['head'])
            ''' Aspects '''
            for aspect in obj['aspects']:
                if aspect['term'] == 'NULL': # For Restaurant 16 dataset
                    new_aspect = {'term': ['NULL'], 'from': str(0), 'to': str(0), 'head': new_obj['head'], 'deprel': new_obj['deprel'], 'path': list(), 'polarity': aspect['polarity']}
                else:
                    new_aspect = {'term': list(), 'path': list(), 'polarity': aspect['polarity']}
                    ''' From and To indices '''
                    begin = False
                    for i in range(len(new_obj['token'])):
                        if (not begin) and char_index[i] >= int(aspect['from']):
                            new_aspect['from'] = str(i)
                            begin = True
                        if char_index[i] >= int(aspect['to']):
                            new_aspect['to'] = str(i)
                            begin = False
                            break
                        if begin:
                            new_aspect['term'].append(new_obj['token'][i])
                    if begin:
                        new_aspect['to'] = str(len(new_obj['token']))
                    assert int(new_aspect['from']) < int(new_aspect['to'])
                    ''' Aspect Head and Dependency '''
                    new_aspect['head'] = [0 for i in range(len(new_obj['token']))]
                    new_aspect['deprel'] = ['ROOT' for i in range(len(new_obj['token']))]
                    node_flag = [True for i in range(len(new_obj['token']))]
                    def transfrom(node):
                        node_flag[node] = False
                        for j in connect_nodes[node]:
                            if node_flag[j]:
                                new_aspect['head'][j] = node+1
                                if new_aspect['head'][j] != new_obj['head'][j]:
                                    new_aspect['deprel'][j] = f"rev#{new_obj['deprel'][node]}"
                                else:
                                    new_aspect['deprel'][j] = new_obj['deprel'][j]
                                transfrom(j)
                    transfrom(int(new_aspect['from']))
                ''' Aspect Paths '''
                for i in range(len(new_obj['token'])):
                    new_aspect['path'].append(list())
                    def dfs(r):
                        if new_aspect['head'][r]:
                            new_aspect['path'][i].insert(0, r+1)
                            dfs(new_aspect['head'][r]-1)
                    dfs(i)
                new_obj['aspects'].append(new_aspect)
            succeed_objs.append(new_obj)
            ratio = int((k+1)*50/obj_num)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {k+1}/{obj_num} {(k+1)*100/obj_num:.2f}%")
            sys.stdout.flush()
        except Exception:
            print(f"\nParse error at {obj['raw_text']}")
            raise
    print()
    return succeed_objs

def countObj(objs):
    polarity_dict = {'positive': 0, 'negative': 0, 'neutral': 0}
    for obj in objs:
        for aspect in obj['aspects']:
            polarity_dict[aspect['polarity']] += 1
    return polarity_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', '-f', default='Restaurants_Trial,Restaurants_Train,Restaurants_Test,Laptops_Trial,Laptops_Train,Laptops_Test,Tweets_Train,Tweets_Test,Restaurants16_Trial,Restaurants16_Train,Restaurants16_Test', type=str, help='FILE NAME')
    parser.add_argument('--corenlp', '-c', default='D:/StanfordCorenlp/stanford-corenlp-4.0.0', type=str, help='CORENLP DIR')
    opt = parser.parse_args()
    os.environ["CORENLP_HOME"] = opt.corenlp
    if not os.path.exists('./datasets/parsed'):
        os.mkdir('./datasets/parsed')
    fns = [f.strip() for f in opt.file_list.split(",")]
    log = list()
    with corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit', 'depparse'], properties={'ssplit.isOneSentence': True}, memory='8G', timeout=30000) as client:
        for fn in fns:
            msg = f"{fn}.xml is processing..."
            print(msg)
            log.append(msg)
            objs = parseXML(f"./datasets/raw/{fn}.xml")
            succeed = parseSentences(objs, client)
            num = countObj(succeed)
            with open(f"./datasets/parsed/{fn}.json", 'w', encoding='utf-8') as f:
                f.write(json.dumps(succeed, sort_keys=False, indent=4))
            msg = f"Processed {len(succeed)} instances, total {sum(num.values())} aspects, {num['positive']} positive, {num['negative']} negative, and {num['neutral']} neutral."
            print(msg)
            log.append(msg)
    with open('./datasets/lastest.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))
             