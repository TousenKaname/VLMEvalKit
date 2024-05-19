import os.path as osp
import pandas as pd
from tqdm import tqdm
from vlmeval.evaluate.misc import build_judge
from vlmeval.utils import can_infer, track_progress_rich, TSVDataset
from vlmeval.smp import *
import numpy as np

INTERNAL = os.environ.get('INTERNAL', 0)

abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}


def report_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'l2-category', 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = abbrs[ab] if ab in abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    return pd.DataFrame(res)


def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant specifically designed to help match the answer to a question with the provided options.'
        'In this task, you will get: a specific question, several different options, and multiple answers. Your main task is to identify which options are most closely related to the answer.'
        'Note: Please note the capital letters in your answers; If the answer already meets the requirements, has only uppercase letters, and is separated by ",", the output is as is.'
        'What you need to output is the uppercase label for the option that matches the answer, separated by commas ",", and in alphabetical order, such as "A,B,C", "B,C", "A,C,D", "A,B,C,D" (if these are the matching options), or "Z". Be careful not to print out superfluous text.'
        'Example 1:'
        'Question: What is most suitable for outdoor activities in summer?\nOptions: A. Sunbathing B. Skiing C. Bonfire D. Swimming pool\nAnswer: Swimming and sunbathing\nOutput: A,D\n'
        'Example 2: \n'
        'Question: From the options given, select the part that best matches the image. Option: A. reading A book B. fishing C. running D. playing cards E. Planting trees\n Answer: The most matched part of the image is A,B,C\n Output: A,B,C\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nOutput: '
    )
    return tmpl.format(question, options, prediction)


def build_prompt_cn(question, options, prediction):
    tmpl = (
        '你是一个专门帮助匹配问题答案与提供的选项的 AI 助手。'
        '在这个任务中，你将获得：一个具体问题，几个不同的选项，以及多个答案。你的主要任务是识别出哪些选项与答案内容最为接近。'
        '提示：请你注意答案中的大写字母；如果答案已经满足要求，只有大写字母，并且用","分开，则原样输出'
        '你需要输出的是与答案匹配的选项的大写字母标签，字母之间用逗号 "," 隔开，按照字母顺序排列，如 "A,B,C"、"B,C"、"A,C,D"、"A,B,C,D"（如果这些是符合的选项），或者 "Z"。请注意，不要输出多余的文本。'
        '例 1:'
        '问题: 最适合夏日户外活动的是什么?\n选项: A. 日光浴 B. 滑雪 C. 篝火 D. 泳池\n答案: 游泳和日光浴\n输出: A,D\n'
        '例 2: \n'
        '问题: 从给出的选项中，选择与图像中最匹配的部分。\n选项: A. 看书 B. 钓鱼 C. 跑步 D. 打牌 E.植树\n答案: 图像中最匹配的部分是A、B、C\n输出: A,B,C\n'
        '例 3: \n'
        '问题: {}?\n选项: {}\n答案: {}\n输出: '
    )
    return tmpl.format(question, options, prediction)


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

def extract_answer_from_item(model, item):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3

    while retry:
        ans = model.generate(prompt).strip()
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
        else:
            return dict(opt=ans, log=ans)
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            return dict(opt=rd.choice(options), log='Failed to predict, thus randomly generate one. ')


def eval_sub_data(model, sub_data, answer_map):
    lt = len(sub_data)
    log = ''
    for i in range(lt):
        res = extract_answer_from_item(model, sub_data.iloc[i])
        opt, match_log = res['opt'].strip(), res['log'].strip()

        opt = ''.join(sorted(set(opt.replace(',', ''))))
        index = sub_data.iloc[i]['index']  # 使用 'index' 列作为唯一标识符
        answer_map[index] = ''.join(sorted(set(answer_map[index].replace(',', ''))))
        

        if opt != answer_map[index]:
            log += (
                f"Failed in Rolling {i}: Answer is {answer_map[index]}; Prediction is {sub_data.iloc[i]['prediction']}; "
                f'Pre-fetched is {opt}; Match Log is {match_log}.\n'
            )
            return dict(hit=0, log=log)
        else:
            log += (
                f"Rolling {i}: Answer is {answer_map[index]}, Prediction is {sub_data.iloc[i]['prediction']}, "
                f'Pre-fetched is {opt}.\n'
            )

    return dict(hit=1, log=log)


def eval_data_groups(model, data_groups, answer_map, result, result_file, nproc=16):
    remain = []
    for dg in data_groups:
        remain.append(dg)
    tups = [(model, x, answer_map) for x in remain]
    keys = [x.iloc[0]['index'] % 1e6 for x in remain]
    if len(tups) == 0:
        return

    res = track_progress_rich(
        eval_sub_data,
        tups,
        nproc=nproc,
        chunksize=nproc,
        save=result_file,
        keys=keys)
    result = load(result_file)
    for k, v in zip(keys, res):
        if k in result:
            assert result[k]['hit'] == v['hit'] and result[k]['log'] == v['log']
        else:
            result[k] = v
    dump(result, result_file)


def multiple_response_eval(eval_file, dataset='default', **judge_kwargs):
    logger = get_logger('Evaluation')

    # assert dataset is not None
    nproc = judge_kwargs.pop('nproc', 4)

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    model = judge_kwargs['model']
    assert model in ['chatgpt-0613', 'exact_matching', 'gpt-4-0125']
    name_str_map = {
        'chatgpt-0613': 'openai',
        'gpt-4-0125': 'gpt4'
    }
    name_str = name_str_map[model] if model in name_str_map else model


    if INTERNAL or gpt_key_set():
        model = build_judge(**judge_kwargs)
    else:
        raise ValueError('multi-response only support LLM-based answer extraction')

    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    data = load(eval_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    if dataset != 'default':
        meta = TSVDataset(dataset).data
    else:
        logger.warning('Dataset is not provided, try to use the original `eval_file` as meta data. ')
        meta = load(eval_file)
        assert 'index' in meta and 'answer' in meta, 'Essentail columns missing in the eval_file.'

    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])} if 'category' in meta else None
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])} if 'l2-category' in meta else None
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])} if 'split' in meta else None

    if cate_map is not None and np.all([pd.isna(x) for x in cate_map.values()]):
        cate_map = None
    if l2_cate_map is not None and np.all([pd.isna(x) for x in l2_cate_map.values()]):
        l2_cate_map = None
    if split_map is not None and np.all([pd.isna(x) for x in split_map.values()]):
        split_map = None

    data = data[data['index'].isin(answer_map)]
    data_main = data[data['index'] < int(1e6)]
    meta_idx_set = set(meta['index'])
    data_main = data_main[data_main['index'].isin(meta_idx_set)]

    lt = len(data_main)
    hit, tot = 0, 0

    data_groups = []
    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']

        if idx in result:
            correct = result[idx]['hit']
            assert correct in [0, 1]
            hit += correct
            tot += 1
            continue

        sub_data = data[data['index'] % int(1e6) == idx]
        data_groups.append(sub_data)

    if len(data_groups):
        eval_data_groups(
            model=model,
            data_groups=data_groups,
            answer_map=answer_map,
            nproc=nproc,
            result=result,
            result_file=result_file)

    tmp_pth = f'/tmp/{timestr()}.xlsx'
    dump(data_main, tmp_pth)
    data_main = load(tmp_pth)

    res = load(result_file)
    indices = data_main['index']

    data_main['hit'] = [res[i]['hit'] for i in indices]
    data_main['log'] = [res[i]['log'] for i in indices]

    main_idx = data_main['index']
    if cate_map is not None:
        data_main['category'] = [cate_map[i] for i in main_idx]
    if l2_cate_map is not None:
        data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
    if split_map is not None:
        data_main['split'] = [split_map[i] for i in indices]

    # load split
    dump(data_main, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    data_main = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

    acc = report_acc(data_main)
    score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
    dump(acc, score_file)
    logger.info(f'multiple_response_eval successfully finished evaluating {eval_file}, results saved in {score_file}')
    logger.info('Score: ')
    logger.info(acc)
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM Answers. ')
    parser.add_argument('data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument(
        '--model',
        type=str,
        help='The LLM (GPT) used for inference. ',
        default='chatgpt-0613',
        choices=['chatgpt-0613', 'exact_matching', 'gpt-4-0125'])
    parser.add_argument(
        '--dataset',
        type=str,
        default='default',
        help='The dataset to evaluate')
    parser.add_argument('--nproc', type=int, default=6)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    acc = multiple_response_eval(
        eval_file=args.data, model=args.model, dataset=args.dataset, nproc=args.nproc, verbose=args.verbose)
