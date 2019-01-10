import evaluate_result
from utils import *
import ConvertFeatures
import TrainSolver
import Predict

save_feature_here = "memm-features"
file_name = "data/Corpus.TRAIN.txt"
dev_ann = "data/TRAIN.annotations"
processed_file_name = "data/Corpus.TRAIN.processed"
file_name_for_all_ner = "all_ner_file_dict.pickle"
stanford_ner_pickle = "stnaford_ner.pickle"
combind_sentences_pickle = "combined_dict.pickle"
Mr_Mrs = set(['Mrs.', 'Ms.'])
DEBUG = False

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


def find_closest_location(all_location, person_location):
    for i, loc in enumerate(all_location):
        location_index = loc[1]
        if i == 0 or abs(location_index - person_location) < this_min:
            this_min = abs(location_index - person_location)
            arg_min = loc
    return arg_min


def main():
    wrost_case = 0
    preson_twich = 0
    location_twich = 0


    correct_annotations = get_tags_from_annotations(dev_ann)

    if (load_from_pickle):
        all_stanford_text = load_from_file(stanford_ner_pickle)
    else:
        all_stanford_text = {}

    tokens_sentences = convert_sentences_to_tokens(processed_file_name)
    processed_dict = processed_text_to_dict(processed_file_name)
    all_txt = []
    false_line = []
    fal = pos = 0
    word_to_route, all_sentence_data = get_path_from_word(processed_file_name)
    with open(file_name) as f:
        for i, line in enumerate(f):
            line = line.split("\t")
            sen_num = line[0]
            route_to_root = word_to_route[sen_num]
            if (load_from_pickle):
                stanford = all_stanford_text[sen_num]
            else:
                sen = tokens_sentences[sen_num]
                stanford = stanford_extract_ner_from_sen(sen)
                all_stanford_text[sen_num] = stanford

            this_sentence_proccesed_data = all_sentence_data[sen_num]
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(), processed_dict[sen_num],this_sentence_proccesed_data)

            ners = extract_ner(combine_processed_and_stanford)
            person_location_ner = check_person_and_location(ners)

            if not (person in person_location_ner and location in person_location_ner):
                continue
            possiable_persons, possiable_location = unique_person_and_location(person_location_ner[person], person_location_ner[location])
            # for k_p,p in possiable_persons.items():
            #     for k_p,l in possiable_location.items():
            #         if len(p) > 1 and len(l) > 1:
            #             wrost_case +=1
            #         elif len(p) > 1:
            #             preson_twich +=1
            #         elif len(l) > 1:
            #             location_twich += 1
            #             print(line[1])
            # print(wrost_case, preson_twich, location_twich)

            for per in possiable_persons:
                for loc in possiable_location:
                    per_tup, loc_tup = create_nereast_tupple(per,possiable_persons[per],loc,possiable_location[loc])
                    feature = extract_feature(per_tup, loc_tup, route_to_root, [possiable_persons[per], possiable_location[loc]],this_sentence_proccesed_data)
                    true_or_not = tupple_in_annotion(per_tup, loc_tup, correct_annotations[sen_num])
                    if (DEBUG and len(possiable_persons) * len(possiable_location) == 1):
                        fal += true_or_not == 0
                        pos += true_or_not == 1
                        # print(sen_num)
                        if (not true_or_not):
                            false_line.append(line)
                    txt = convert_to_text(true_or_not, feature)
                    all_txt.append(txt)
    if (DEBUG):
        print("pos ", pos)
        print("fal ", fal)
        for p in false_line:
            print(p)
    write_to_file(save_feature_here, all_txt)
    features_vec_file, features_map_file = ConvertFeatures.main(save_feature_here)
    model_file = TrainSolver.main(features_vec_file)
    output_file_name = "SVM_OUTPUT.txt"
    clean_input_file_name = "data/Corpus.DEV.txt"
    Predict.main(clean_input_file_name=clean_input_file_name, model_filename=model_file,
                 feature_map_filename=features_map_file, output_file_name=output_file_name)
    evaluate_result.main(output_file_name, "data/DEV.annotations")


if __name__ == '__main__':
    main()
