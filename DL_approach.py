import numpy as np
from navie_approach import *
from nltk.tag.stanford import StanfordNERTagger
import evaluate_result
st = StanfordNERTagger('/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
               '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/stanford-ner.jar')

file_name = "data/Corpus.TRAIN.txt"
processed_file_name = "data/Corpus.TRAIN.processed"
person = 'PERSON'
location = 'LOCATION'
file_name_for_all_ner = "all_ner_file_dict.pickle"
pred_file_output = "DL_approach_output.txt"

def save_to_file(var , file_name):
    print (var)
    with open(file_name, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_vectors(file_name):
    word_to_index_local = {}
    list_of_vec = []
    with open(file_name) as f:
        for i,line in enumerate(f):
            parts = line.split()
            word  = parts[0]
            vec   = parts[1:]
            word_to_index_local[word] = len(word_to_index_local)
            list_of_vec.append(np.array(vec, dtype=np.float64))

    return np.matrix(list_of_vec) , word_to_index_local


def main():
    tokens_sentences = convert_sentences_to_tokens(processed_file_name)
    processed_dict = processed_text_to_dict(processed_file_name)
    all_stanford_text= {}
    with open(file_name) as f:
        save_all_text = []
        all_sentence_ner_dict = {}
        all_sentence_ner_dict = load_from_file(file_name_for_all_ner)
        all_stanford_text   = load_from_file(stanford_ner_pickle)
        for i,line in enumerate(f):
            print(i)
            line = line.split("\t")
            sen_num = line[0]
            # sen = tokens_sentences[sen_num]
            stanford = all_stanford_text[sen_num]
            # stanford = stanford_extract_ner_from_sen(sen)
            # all_stanford_text[sen_num] = stanford
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(),processed_dict[sen_num])
            ners = extract_ner(combine_processed_and_stanford)
            ner_dict = check_person_and_location(ners)
            all_sentence_ner_dict[sen_num] = ner_dict

            text = sen_num + "\t"
            ner_dict =  all_sentence_ner_dict[line[0]]
            # if sen_num == 'sent119':
            #     stanford = stanford_extract_ner_from_sen(tokens_sentences[sen_num])
            #     combine_processed_and_stanford = combine_two_sentences(stanford, processed_dict[sen_num])
            #     ners = extract_ner(combine_processed_and_stanford)
            #     ner_dict = check_person_and_location(ners)
            #     print(1)

            if not (person in ner_dict and location in ner_dict):
                continue


            possiable_persons, possiable_location = unique_person_and_location(ner_dict[person],ner_dict[location])
            if sen_num == 'sent119':
                print(1)
            for per in possiable_persons:
                person_appread_in = per[1]
                # loc = find_closest_location(ner_dict[location],person_appread_in)
                for loc in possiable_location:
                #     loc_appread_in = loc[1]
                    # if (len(ner_dict[person]) == 1 and len(ner_dict[location]) == 1):
                # if abs(loc_appread_in - person_appread_in) < 1500:
                    text_line = text + per[0] + "\tLive_In\t" + loc[0] + "\n"
                    save_all_text.append(text_line)

    write_to_file(pred_file_output, save_all_text)
    evaluate_result.main(pred_file_output)
    # save_to_file(all_sentence_ner_dict, file_name_for_all_ner)
    # save_to_file(all_stanford_text,stanford_ner_pickle)

if __name__ == '__main__':
   main()
