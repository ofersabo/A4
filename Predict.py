import numpy as np
import time
import sys
from utils import *
import pickle
import numpy as np
from codecs import open
import scipy
DEV_STANFORD_NER = "DEV_STANFORD_NER"
all_states = set()
skip_start = 0
end_early = -1
model = None
from collections import Counter

unigram = {}
bigram = {}
feature_dict = {}
all_tags_dict = {}
index_to_tag = {}
deno = 'denominator'
not_rare_words = set()
temp_file = "files/temp"


def build_matrix(feature_index,num_feature):
    features_matrix = scipy.sparse.lil_matrix((1, num_feature))
    for feature_id in feature_index:
        features_matrix[0, feature_id] = 1

    return features_matrix

def list_of_index(line,feature_dict,outside):
    feature_index_per_word = []
    features = line.strip().split(" ")
    string_of_line = "0"

    for f in features:
        if f in feature_dict:
            feature_index_per_word.append(feature_dict[f])
        else:
            outside.append(f)
    feature_index_per_word = sorted(feature_index_per_word)
    return string_of_line, feature_index_per_word


def predict(matrix, clf, num_feature):
    value = clf.predict(matrix)
    return value


def load_model(model_filename):
    loaded_model = pickle.load(open(model_filename, 'rb'))
    return loaded_model


def convert_to_vec(txt,outside):
    tag = 0
    tag_str ,feature_index = list_of_index(txt,feature_dict,outside)
    matrix = build_matrix(feature_index,len(feature_dict))
    res = predict(matrix,model,len(feature_dict))[0]
    return res




def find_answer(clean_input_file_name,proccessed_input_file_name, output_file_name, golden_file):
    correct_annotations = get_tags_from_annotations(golden_file)
    # combined_dict = load_from_file(combind_sentences_pickle)
    # all_sentence_ner_dict = load_from_file(file_name_for_all_ner)
    all_stanford_text = load_from_file(DEV_STANFORD_NER)
    # all_stanford_text = {}
    outside = []
    save_all_text = []
    processed_dict = processed_text_to_dict(proccessed_input_file_name)
    combined_dict = {}
    word_to_route, all_sentence_data = get_path_from_word(proccessed_input_file_name)
    with open(clean_input_file_name) as f:
        all_sentence_ner_dict = {}
        for i, line in enumerate(f):
            line = line.split("\t")
            sen_num = line[0]
            route_to_root = word_to_route[sen_num]
            this_sentence = all_sentence_data[sen_num]
            sen = [k[1] for k in this_sentence]
            # stanford = stanford_extract_ner_from_sen(sen)
            stanford = all_stanford_text[sen_num]
            # stanford = stanford_extract_ner_from_sen(sen)
            #all_stanford_text[sen_num] = stanford

            this_sentence_proccesed_data = all_sentence_data[sen_num]
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(), processed_dict[sen_num],this_sentence_proccesed_data)
            combined_dict[sen_num] = combine_processed_and_stanford

            ners = extract_ner(combine_processed_and_stanford)
            ner_dict = check_person_and_location(ners)
            all_sentence_ner_dict[sen_num] = ners

            text = sen_num + "\t"
            #ner_dict = all_sentence_ner_dict[sen_num]

            if not (person in ner_dict and location in ner_dict):
                continue

            possiable_persons, possiable_location = unique_person_and_location(ner_dict[person], ner_dict[location])
            for per in possiable_persons:
                for loc in possiable_location:
                    feature = extract_feature(per, loc, route_to_root, [possiable_persons, possiable_location],this_sentence_proccesed_data)
                    # feature.append((len(possiable_persons))*len(possiable_location)==1)

                    # feature.append((len(possiable_persons)))
                    # feature.append((len(possiable_location)))
                    true_or_not = tupple_in_annotion(per, loc, correct_annotations[sen_num])
                    txt = convert_to_text_only_feature(feature)
                    pred = convert_to_vec(txt,outside)
                    if pred: #or len(possiable_persons)*len(possiable_location)==1:
                        text_line = text + per[0] + "\tLive_In\t" + loc[0] + "\n"
                        save_all_text.append(text_line)

    write_to_file(output_file_name, save_all_text)
    save_to_file(all_stanford_text,DEV_STANFORD_NER )
    return all_sentence_ner_dict

def analyze_feature_map(input_file):
    f = open(input_file, "r")
    for i, line in enumerate(f):
        parts = line.strip().split(" ")
        tag = parts[0]
        index = parts[1]
        feature_dict[tag] = int(index)


def main(clean_input_file_name = "data/Corpus.DEV.txt",input_file_name="data/Corpus.DEV.processed", model_filename="saved_model_short",
         feature_map_filename="feature_map_file.txt", output_file_name = "SVM_OUTPUT.txt", golden_file="data/DEV.annotations"):
    # input_file_name q_mle_filename e_mle_filename output_file_name extra_file_name
    global model
    model = load_model(model_filename)
    analyze_feature_map(feature_map_filename)
    all_sentence_ner_dict = find_answer(clean_input_file_name,input_file_name, output_file_name, golden_file)  # ../files/MEMM_output.txt
    return output_file_name,all_sentence_ner_dict

if __name__ == '__main__':
    import evaluate_result
    start = time.time()
    output_file_name,all_sentence_ner_dict = main()
    missd_rel = evaluate_result.main(output_file_name, "data/DEV.annotations")
    missed_locs = 0
    missed_pers = 0
    for miss in missd_rel:
        sen_num = miss[0]
        ner = all_sentence_ner_dict[sen_num]
        org_list = [item[0] for item in ner if item[1] == "ORGANIZATION"]
        if miss[1] in org_list:
            missed_pers += 1
        if miss[2] in org_list:
            missed_locs += 1
    print(missed_pers)
    print(missed_locs)





    end = time.time()
    print
    "time is %f" % (end - start)
