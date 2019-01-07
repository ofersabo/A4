import pickle
from nltk.tag.stanford import StanfordNERTagger
import evaluate_result
from utils import *
import ConvertFeatures
import TrainSolver
import Predict

st = StanfordNERTagger(
    '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
    '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/stanford-ner.jar')

save_feature_here = "memm-features"
file_name = "data/Corpus.TRAIN.txt"
dev_ann = "data/TRAIN.annotations"
processed_file_name = "data/Corpus.TRAIN.processed"
person = 'PERSON'
location = 'LOCATION'
file_name_for_all_ner = "all_ner_file_dict.pickle"
stanford_ner_pickle = "stnaford_ner.pickle"
combind_sentences_pickle = "combined_dict.pickle"
Mr_Mrs = set(['Mrs.', 'Ms.'])


def save_to_file(var, file_name):
    print(var)
    with open(file_name, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_file(file_name):
    with open(file_name, 'rb') as f:
        var = pickle.load(f)
    return var


def convert_sentences_to_tokens(file_name):
    sentences = {}
    last_line_is_blank = True
    for i, line in enumerate(open(file_name)):
        line = line.strip().replace("\t", " ").split()
        if last_line_is_blank:
            last_line_is_blank = False
            sen_num = line[-1]
            sentences[sen_num] = []
            this_sentence = sentences[sen_num]
            continue
        elif len(line) == 0:
            last_line_is_blank = True
            continue
        elif line[0].isdigit():
            word = line[1]
            this_sentence.append(word)

    return sentences


def processed_text_to_dict(file_name):
    sentence_dict = {}
    last_line_is_blank = True
    for i, line in enumerate(open(file_name)):
        line = line.strip().replace("\t", " ").split()
        if last_line_is_blank:
            last_line_is_blank = False
            sen_num = line[-1]
            sentence_dict[sen_num] = sentence_dict.get(sen_num, list())
            this_sentence = sentence_dict[sen_num]
            continue
        elif len(line) == 0:
            last_line_is_blank = True
            continue
        elif line[0].isdigit():
            word = line[1]
            ner = line[-1]
            if ner == "GPE" or ner == "NORP": ner = "LOCATION"
            # if ner == "GPE": ner = "LOCATION"
            if line[3] == 'POS':
                ner = 'O'
            if ner == person and len(this_sentence) > 0:
                pre_tup = this_sentence[-1]
                pre_word = pre_tup[0]
                if (pre_word == 'Mrs.' or pre_word == 'Ms.'):
                    this_sentence[-1] = (pre_word, person)

            this_sentence.append((word, ner))

    return sentence_dict


def tupple_to_file(file_name, list_of_tupples):
    with open(file_name, 'w') as f:
        for s in list_of_tupples:
            for t in s:
                sr = t[0] + "\t" + t[1] + "\n"
                f.write(sr)


def stanford_extract_ner_from_sen(sen):
    r = st.tag(sen)
    return r


def find_closest_location(all_location, person_location):
    for i, loc in enumerate(all_location):
        location_index = loc[1]
        if i == 0 or abs(location_index - person_location) < this_min:
            this_min = abs(location_index - person_location)
            arg_min = loc
    return arg_min


def main():
    correct_annotations = get_tags_from_annotations(dev_ann)
    combined_dict = load_from_file(combind_sentences_pickle)
    tokens_sentences = convert_sentences_to_tokens(processed_file_name)
    processed_dict = processed_text_to_dict(processed_file_name)
    all_stanford_text = {}
    all_txt = []
    false_line= []
    fal = pos = 0
    word_to_route, all_sentence_data = get_path_from_word(processed_file_name)
    with open(file_name) as f:
        save_all_text = []
        all_sentence_ner_dict = {}
        all_sentence_ner_dict = load_from_file(file_name_for_all_ner)
        all_stanford_text = load_from_file(stanford_ner_pickle)
        for i, line in enumerate(f):
            print(i)
            line = line.split("\t")
            sen_num = line[0]
            route_to_root = word_to_route[sen_num]
            # sen = tokens_sentences[sen_num]
            stanford = all_stanford_text[sen_num]
            # stanford = stanford_extract_ner_from_sen(sen)
            # all_stanford_text[sen_num] = stanford
            this_sentence_proccesed_data = all_sentence_data[sen_num]
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(), processed_dict[sen_num],this_sentence_proccesed_data)
            combined_dict[sen_num] = combine_processed_and_stanford



            ners = extract_ner(combine_processed_and_stanford)
            ner_dict = check_person_and_location(ners)
            all_sentence_ner_dict[sen_num] = ner_dict

            text = sen_num + "\t"
            ner_dict = all_sentence_ner_dict[line[0]]

            if not (person in ner_dict and location in ner_dict):
                continue

            possiable_persons, possiable_location = unique_person_and_location(ner_dict[person], ner_dict[location])
            for per in possiable_persons:
                for loc in possiable_location:
                    feature = extract_feature(per, loc, route_to_root, combine_processed_and_stanford,
                                              this_sentence_proccesed_data)
                    # feature.append((len(possiable_persons))*len(possiable_location)==1)
                    # feature.append((len(possiable_location)))
                    true_or_not = tupple_in_annotion(per, loc, correct_annotations[sen_num])
                    if (len(possiable_persons)*len(possiable_location)==1):
                        fal +=  true_or_not == 0
                        pos +=  true_or_not == 1
                        print(sen_num)
                        if (not true_or_not):
                            false_line.append(line)
                    txt = convert_to_text(true_or_not, feature)
                    all_txt.append(txt)
    print("pos ",pos)
    print("fal ",fal)
    for p in false_line:
        print(p)
    write_to_file(save_feature_here, all_txt)
    features_vec_file, features_map_file = ConvertFeatures.main(save_feature_here)
    model_file = TrainSolver.main(features_vec_file)
    output_file_name = "SVM_OUTPUT.txt"
    clean_input_file_name = "data/Corpus.DEV.txt"
    Predict.main(clean_input_file_name=clean_input_file_name,model_filename=model_file, feature_map_filename=features_map_file, output_file_name=output_file_name)
    evaluate_result.main(output_file_name,"data/DEV.annotations")

if __name__ == '__main__':
    main()
