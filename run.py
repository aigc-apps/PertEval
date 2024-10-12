import argparse
import json

from datetime import datetime
from functools import partial

import dataset_preprocessing as dps
from GeneralLLM import LargeLanguageModel, ChatGPT, Qwen, GLM, Gemini
import KPPerturbation as kpp
from run_benchmark import test_dataset, parallel_test_dataset
import transition_analysis as tas

def get_llm_from_name(name:str, api_key:str)->LargeLanguageModel:
    if "gpt-" in name:
        return ChatGPT(name = name, model = name, api_key = api_key)

    elif "qwen-" in name:
        return Qwen(name = name, model = name, api_key = api_key)

    elif "glm-" in name:
        return GLM(name = name, model = name, api_key = api_key)

    elif "gemini-" in name:
        return Gemini(name = name, model = name, api_key = api_key)

    return -1    

def get_llm_class_from_name(name:str, api_key:str)->LargeLanguageModel:
    if "gpt-" in name:
        return partial(ChatGPT, api_key = api_key)

    elif "qwen-" in name:
        return partial(Qwen, api_key = api_key)

    elif "glm-" in name:
        return partial(GLM, api_key = api_key)

    elif "gemini-" in name:
        return partial(Gemini, api_key = api_key)

    return -1    

def get_pert_from_config(config:dict):
    meta = config['meta_pert']
    pert_names = config['pert']['perturbation']
    atom_perts = []
    current = None
    for elem in pert_names:
        if elem == "KnInvPara":
            current = kpp.ParaphrasingPerturbation(
                paraphrase_config = {
                    'n_candidates':meta['KnInvPara']['n_candidates'],
                    'similarity_score':meta['KnInvPara']['similarity_score']},
                rewriter = get_llm_from_name(
                    meta['KnInvPara']['model'], meta['KnInvPara']['api_key']))

        elif elem == "OptionPerm":
            current = kpp.OptionPermutationPerturbation(
                permutation_map = eval(meta['OptionPerm']['permutation_map']))

        elif elem == "OptionForm":
            current = kpp.OptionFormatPerturbation(
                method = meta['OptionForm']['method'])
    
        elif elem == "OptionCaesar":
            current = kpp.CaesarPerturbation(delta = meta['OptionCaesar']['delta'])

        elif elem == "ChangeType":
            current = kpp.ChangeTypePerturbation()

        elif elem == "SwapPos":
            current = kpp.ChangeQuestionPosPerturbation()

        else:
            print(f"Perturbation '{elem}' undefined. Skip.")
            continue
        atom_perts.append(current)
    return kpp.MixedPerturbation(atom_perts)

def get_key_from_config_and_name(config:dict, name:str)->str:
    if "gpt-" in name:
        return config['meta_llm_apis']['openai_key']

    elif "qwen-" in name:
        return config['meta_llm_apis']['qwen_key']

    elif "glm-" in name:
        return config['meta_llm_apis']['glm_key']

    elif "gemini-" in name:
        return config['meta_llm_apis']['gemini_key']

    return -1    
    
def get_parser():
    parser = argparse.ArgumentParser(description = "This is the PertEval toolkit launcher. Before running PertEval, don't forget to transform your evaluation dataset into the targeted .jsonl file (see example.ipynb for data format)")
    parser.add_argument('--action', type=str, required = True, help = "The action you want PertEval do. Options: pert - generate the perturbed data; eval - evaluate model knowledge capacity using perturbed and original data; ki_scoring - do llm-based knowledge-invariant scoring; analysis - analyze model knowledge capacity given log paths.; all - do all of actions above one by one.")
    parser.add_argument('--config_path', type=str, default = "config.json", help = "The path to the PertEval configuration json file. Default - config.json")
    parser.add_argument('--model_for_eval', type=str, help = "The model to be evaluated using PertEval. Only necessary when action = eval or all.")
    parser.add_argument('--source', type=str, help = "The file path of the original .jsonl data.")
    parser.add_argument('--target', type=str, help = "The file path of the perturbed data. It is also the target path for saving new perturbed_data if action = pert or all.")
    parser.add_argument('--log_save', type=str, help = "The prefix of the log path for saving PertEval testing records.")
    parser.add_argument('--log_original', type=str, help = "Only required when action = analysis. It is the log path on the original benchmark.")
    parser.add_argument('--log_perturbed', type=str, help = "Only required when action = analysis. It is the log path on the perturbed benchmark.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_path, 'r') as fp:
        config = json.load(fp)
    
    if args.action == 'pert' or args.action == 'all':
        perturbation = get_pert_from_config(config)
        perturbed_data = dps.perturb_dataset(
            ptb = perturbation,
            file_name = args.source,
            target_name = args.target)

    log1_path, log2_path = None, None
    eval_meta = config['eval']
    if args.action == 'eval' or args.action == 'all':
        # Test for the original dataset
        api_key = get_key_from_config_and_name(config, args.model_for_eval)
        log_path_source = f"{args.log}_original"
        log_path_target = f"{args.log}_perturbed"
        log1_path = parallel_test_dataset(
            file_path = args.source,
            log_path_prefix = log_path_source,
            simple_question_path = None,
            subjects = eval_meta['subjects'],
            model_class = get_llm_class_from_name(
                name = args.model_for_eval,
                api_key = api_key),
            model_selection = args.model_for_eval,
            temperature = eval_meta['temperature'],
            thread_func = test_dataset,
            n_thread = eval_meta['n_thread'],
            start_id = None,
            end_id = None
        )

        log2_path = parallel_test_dataset(
            file_path = args.target,
            log_path_prefix = log_path_target,
            simple_question_path = None,
            subjects = eval_meta['subjects'],
            model_class = get_llm_class_from_name(
                name = args.model_for_eval,
                api_key = api_key),
            model_selection = args.model_for_eval,
            temperature = eval_meta['temperature'],
            thread_func = test_dataset,
            n_thread = eval_meta['n_thread'],
            start_id = None,
            end_id = None
        )

    kis_meta = config['ki_scoring']
    if args.action == 'ki_scoring' or args.action == 'all':
        referee_name = kis_meta['model']
        api_key = get_key_from_config_and_name(config, referee_name)
        referee = get_llm_from_name(referee_name, api_key)
        ki_save_path = f"{args.log_save}_ki_scoring_{datetime.now()}.jsonl"
        
        print('######## Knowledge Invariance Validation ########')
        tas.knowledge_invariance_analysis(
            args.source, args.target,
            subjects = kis_meta['subjects'],
            referee = referee,
            llm_ki_to_save_path = ki_save_path,
            systematic_gap = kis_meta['sampling_gap']
        )
        print()
        
    if args.action == 'analysis' or args.action == 'all':
        if log1_path is None:
            log1_path = args.log_original
        if log2_path is None:
            log2_path = args.log_perturbed
        
        print("######## Overall Analysis ########")
        tas.transition_analysis(
            log1_path, log2_path, subjects = eval_meta['subjects'])
        print("\n######## Original Dataset ########")
        tas.response_pattern_analysis(log1_path)
        print("\n######## Perturbed Dataset ########")
        tas.response_pattern_analysis(log2_path)

if __name__ == "__main__":
    main()
        