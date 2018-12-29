import pickle
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
               '/Users/ofersabo/PycharmProjects/NLP_A4/stanford-ner-2018-10-16/stanford-ner.jar')

file_name = "data/Corpus.TRAIN.txt"
processed_file_name = "data/Corpus.TRAIN.processed"
person = 'PERSON'
location = 'LOCATION'
file_name_for_all_ner = "all_ner_file_dict.pickle"

def save_to_file(var , file_name):
    print (var)
    with open(file_name, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_file(file_name):
    with open(file_name,'rb') as f:
        var = pickle.load(f)
    return var


def convert_sentences_to_tokens(file_name):
    sentences = {}
    last_line_is_blank = True
    for i,line in enumerate(open(file_name)):
        line = line.strip().replace("\t"," ").split()
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
    for i,line in enumerate(open(file_name)):
        line = line.strip().replace("\t"," ").split()
        if last_line_is_blank:
            last_line_is_blank = False
            sen_num = line[-1]
            sentence_dict[sen_num] = sentence_dict.get(sen_num,list())
            this_sentence = sentence_dict[sen_num]
            continue
        elif len(line) == 0:
            last_line_is_blank = True
            continue
        elif line[0].isdigit():
            word = line[1]
            ner = line[-1]
            if ner=="GPE": ner = "LOCATION"
            this_sentence.append((word,ner))

    return sentence_dict

def extract_ner(sen):
    all_ner = []
    i = 0
    while i < len(sen):
        if sen[i][1] != 'O':
            j = i + 1
            while j < len(sen) and sen[j][1] == sen[i][1]:
                j+=1
            ner =  " ".join([sen[k][0] for k in range(i,j)])
            ner_tag = sen[i][1]
            i = j -1
            all_ner.append((ner,ner_tag,i))
        i += 1
    return all_ner


def check_person_and_location(all_ner):
    dict_ner = {}
    for ele in all_ner:
        if ele[1] == person:
            dict_ner[person] = dict_ner.get(person,list())
            dict_ner[person].append((ele[0],ele[2]))
        elif ele[1] == location:
            dict_ner[location] = dict_ner.get(location, list())
            dict_ner[location].append((ele[0],ele[2]))

    return dict_ner


def write_to_file(file_name,list_of_text):
    with open(file_name,'w') as f:
        for s in list_of_text:
            f.write(s)


def stanford_extract_ner_from_sen(sen):
    r = st.tag(sen)
    return r


def combine_two_sentences(first,second):
    assert len(first) == len(second)
    for i in range(len(first)):
        first_tuple = first[i]
        second_tuple = second[i]
        assert first_tuple[0] == second_tuple[0]
        if first_tuple[1] != second_tuple[1]:
            if first_tuple[1] == 'O':
                first[i] = second[i]
    return first

def main():
    tokens_sentences = convert_sentences_to_tokens(processed_file_name)
    processed_dict = processed_text_to_dict(processed_file_name)
    with open(file_name) as f:
        save_all_text = []
        all_sentence_ner_dict = {}
        all_sentence_ner_dict = load_from_file(file_name_for_all_ner)
        for i,line in enumerate(f):
            print(i)
            line = line.split("\t")
            sen_num = line[0]
            # sen = tokens_sentences[sen_num]
            # stanford = stanford_extract_ner_from_sen(sen)
            # combine_processed_and_stanford = combine_two_sentences(stanford,processed_dict[sen_num])
            # ners = extract_ner(combine_processed_and_stanford)
            # ner_dict = check_person_and_location(ners)
            # all_sentence_ner_dict[sen_num] = ner_dict
            text = sen_num + "\t"
            ner_dict =  all_sentence_ner_dict[line[0]]
            if sen_num == 'sent953':
                stanford = stanford_extract_ner_from_sen(tokens_sentences[sen_num])
                combine_processed_and_stanford = combine_two_sentences(stanford, processed_dict[sen_num])
                ners = extract_ner(combine_processed_and_stanford)
                ner_dict = check_person_and_location(ners)
                print(1)

            if not (person in ner_dict and location in ner_dict):
                continue

            for per in ner_dict[person]:
                for loc in ner_dict[location]:
                    person_appread_in = per[1]
                    loc_appread_in = loc[1]
                    # if (len(ner_dict[person]) == 1 and len(ner_dict[location]) == 1):
                    text_line = text + per[0] + "\tLive_In\t" + loc[0] + "\n"
                    save_all_text.append(text_line)

    write_to_file("save_output.txt", save_all_text)
    # save_to_file(all_sentence_ner_dict, file_name_for_all_ner)

if __name__ == '__main__':
   main()


