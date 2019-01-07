import pickle
from nltk.tag.stanford import StanfordNERTagger
import evaluate_result
from utils import *
import ConvertFeatures
import TrainSolver
import Predict

import mlp

st = StanfordNERTagger(
    '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
    '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/stanford-ner.jar')

train_text = "data/Corpus.TRAIN.txt"
dev_text = "data/Corpus.DEV.txt"
processed_train = "data/Corpus.TRAIN.processed"
processed_dev = "data/Corpus.DEV.processed"
ann_train = "data/TRAIN.annotations"
ann_dev = "data/DEV.annotations"
person = 'PERSON'
location = 'LOCATION'
file_name_for_all_ner = "all_ner_file_dict.pickle"
stanford_TRAIN_ner_pickle = "stnaford_ner.pickle"
stanford_DEV_ner_pickle = "DEV_STANFORD_NER"
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
            if line[3] == 'POS':
                ner = 'O'
            if ner == PERSON and len(this_sentence) > 0:
                pre_tup = this_sentence[-1]
                pre_word = pre_tup[0]
                if (pre_word == 'Mrs.' or pre_word == 'Ms.'):
                    this_sentence[-1] = (pre_word, PERSON)

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


def generate_txt_file(order,pred):
    pred[10]= 1
    all_text = []
    for i,line in enumerate(order):
        if pred[i]:
            text = order[0] + order[1] +"\t" +"Live_In" + "\t" +order[2] +"\n"
            all_text.append(text)
    write_to_file(all_text,)

def find_closest_location(all_location, person_location):
    for i, loc in enumerate(all_location):
        location_index = loc[1]
        if i == 0 or abs(location_index - person_location) < this_min:
            this_min = abs(location_index - person_location)
            arg_min = loc
    return arg_min


def prepare_sentence_with_mask(sentene_with_ner, per, loc):
    person_mask = replace_ner_with_sentnce(sentene_with_ner, per)
    location_mask = replace_ner_with_sentnce(sentene_with_ner, loc)
    person_mask = ' '.join(person_mask)
    location_mask = ' '.join(location_mask)
    return person_mask, location_mask


def replace_ner_with_sentnce(sentene_with_ner, ner_to_replace):
    index = ner_to_replace[1] - 1
    ner_length = len(ner_to_replace[0].split()) - 1
    string_mask = [w[0] for w in sentene_with_ner]
    string_mask.insert(index, '***mask***')
    string_mask = string_mask[0:index + 1] + string_mask[index + 2 + ner_length:]

    return string_mask


def extract_standford_ner(file):
    all_stanford_text = {}
    for i, line in enumerate(open(file)):
        line = line.split("\t")
        sen_num = line[0]
        sen = line[1].split()
        stanford = stanford_extract_ner_from_sen(sen)
        all_stanford_text[sen_num] = stanford
    print ("Done extracting ner from_standord")
    return all_stanford_text


def get_standofrd_ner(stanford_ner_pickle, txt_file):
    if not (stanford_ner_pickle is None):
        all_stanford_text = load_from_file(stanford_ner_pickle)
    else:
        all_stanford_text = extract_standford_ner(txt_file)
        save_to_file(all_stanford_text, stanford_ner_pickle)
    return all_stanford_text


def prepare_data(processed_file,txt_file, stanford_ner_pickle=None,ann = "a"):
    import Bert
    all_stanford_text = get_standofrd_ner(stanford_ner_pickle, txt_file)
    correct_annotations = get_tags_from_annotations(ann)
    processed_dict = processed_text_to_dict(processed_file)
    data = []
    order_data = []
    with open(txt_file) as f:
        for i, line in enumerate(f):
            print(i)
            line = line.split("\t")
            sen_num = line[0]
            stanford = all_stanford_text[sen_num]
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(), processed_dict[sen_num])
            ners = extract_ner(combine_processed_and_stanford)
            person_location_ner = check_person_and_location(ners)

            text = sen_num + "\t"
            if not (PERSON in person_location_ner and LOCATION in person_location_ner):
                continue

            possible_persons, possible_location = unique_person_and_location(person_location_ner[PERSON],
                                                                             person_location_ner[LOCATION])
            for per in possible_persons:
                for loc in possible_location:
                    true_or_not = tupple_in_annotion(per, loc, correct_annotations[sen_num])
                    person_mask, loca_mask = prepare_sentence_with_mask(combine_processed_and_stanford, per, loc)
                    #person_vector, location_vector = Bert.get_vectors_from_bert([person_mask, loca_mask])
                    #data.append((true_or_not, [person_vector, location_vector]))
                    order_data.append((text,per,loc))
    return data,order_data

def main():
    output_file_name =  "DL_OUTPUT.txt"
    first_load = False
    where_to_store_date = "data_for_mlp.pickle"
    if (first_load):
        train,train_order = prepare_data(processed_train,train_text, stanford_TRAIN_ner_pickle,ann_train)
        print("Done prepering train")
        dev,dev_order = prepare_data(processed_dev,dev_text, stanford_DEV_ner_pickle,ann_dev)
        print("Done prepering dev")
        #save_to_file([train,dev],where_to_store_date)
        save_to_file(dev_order,"order")
    else:
        train, dev =load_from_file(where_to_store_date)
        # order =load_from_file("order")


    pred = mlp.train_MLP(train,dev)
    # pred = [0] * len(order)
    generate_txt_file(order,pred,output_file_name)
    evaluate_result.main(output_file_name, "data/DEV.annotations")


if __name__ == '__main__':
    main()
