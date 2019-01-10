import pickle
import evaluate_result
from utils import *

file_name = "data/Corpus.TRAIN.txt"
processed_file_name = "data/Corpus.TRAIN.processed"

file_name_for_all_ner = "all_ner_file_dict.pickle"
stanford_ner_pickle = "stnaford_ner.pickle"
combind_sentences_pickle = "combined_dict.pickle"

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
            dict_ner[person] = dict_ner.get(person, list())
            dict_ner[person].append((ele[0], ele[2]))
        elif ele[1] == location:
            dict_ner[location] = dict_ner.get(location, list())
            dict_ner[location].append((ele[0], ele[2]))

    return dict_ner

def tupple_to_file(file_name,list_of_tupples):
    with open(file_name,'w') as f:
        for s in list_of_tupples:
            for t in s:
                sr = t[0]+"\t" +t[1]+"\n"
                f.write(sr)


def write_to_file(file_name,list_of_text):
    with open(file_name,'w') as f:
        for s in list_of_text:
            f.write(s)





def combine_two_sentences(first,second):
    assert len(first) == len(second)
    for i in range(len(first)):
        first_tuple = first[i]
        second_tuple = second[i]
        assert first_tuple[0] == second_tuple[0]
        if first_tuple[1] != second_tuple[1]:
            if first_tuple[1] == 'O':
                first[i] = second[i]
        if first_tuple[1] == location and i > 0 and first[i - 1][0] in Mr_Mrs:
            first[i - 1][1] = location
    return first


def get_path_from_word(file_name = processed_file_name):
    dict_to_list = list_of_all_sentences_per_word(file_name)
    mega_dict = {}
    for sen in dict_to_list:
        mega_dict[sen] = {}
        this_sen = mega_dict[sen]
        sentence = dict_to_list[sen]
        for w_list in sentence:
            original_word = w_list[1]
            index = w_list[0]
            word = original_word

            con = w_list[6]
            next_index = int(w_list[5])
            route_to_root = []
            while con != 'ROOT':
                route_to_root.append((next_index,con,word))
                con = sentence[next_index-1][6]
                next_index = int(sentence[next_index-1][5])
                word = sentence[next_index-1][1]

            this_sen[original_word] = route_to_root

    return mega_dict


def list_of_all_sentences_per_word(file_name):
    path_dict = {} #dict of dict
    last_line_is_blank = True
    for i,line in enumerate(open(file_name)):
        line = line.strip().replace("\t"," ").split()
        if last_line_is_blank:
            last_line_is_blank = False
            sen_num = line[-1]
            path_dict[sen_num] = []
            this_sentence = path_dict[sen_num]
            continue
        elif len(line) == 0:
            last_line_is_blank = True
            continue
        elif line[0].isdigit():
            this_sentence.append(line)


    return path_dict

def extract_word_from_tuple(tup):
    x = tup[0]
    if " " in x:
        return x.split()[0]
    return x

def find_length_route(per,loc,route_per_word):
    per_word = extract_word_from_tuple(per)
    loc_word = extract_word_from_tuple(loc)
    per_route = route_per_word[per_word]
    loc_route = route_per_word[loc_word]
    if len(set(per_route).intersection(set(loc_route))) == 0:
        return 999

    if len(per_route) > len(loc_route):
        longest = per_route
        shortest = loc_route
    else:
        longest = loc_route
        shortest = per_route

    dis = 0
    route = []
    for i,ele in enumerate(longest,1):
        if ele in set(shortest):
            dis = i
            break
        route.append((ele[1], up))

    for i,ele in enumerate(shortest,1):
        if ele in set(longest):
            dis+=i
            break
        route.append((ele[1], down))
    print(route)
    return dis




def find_closest_location(all_location, person_location):
    for i,loc in enumerate(all_location):
        location_index = loc[1]
        if i == 0 or abs(location_index-person_location) < this_min:
            this_min = abs(location_index-person_location)
            arg_min = loc
    return arg_min

def unique_person_and_location(persons,locations):
    per_set = set()
    person_res = []
    for p in persons:
        if not p[0] in per_set:
            person_res.append(p)
        per_set.add(p[0])

    loc_set = set()
    location_res = []
    for l in locations:
        if not l[0] in loc_set:
            location_res.append(l)
        loc_set.add(l[0])

    return person_res,location_res

def main():
    combined_dict = load_from_file(combind_sentences_pickle)
    tokens_sentences = convert_sentences_to_tokens(processed_file_name)
    processed_dict = processed_text_to_dict(processed_file_name)
    all_stanford_text= {}
    word_to_route  = get_path_from_word(processed_file_name)
    with open(file_name) as f:
        save_all_text = []
        all_sentence_ner_dict = {}
        all_sentence_ner_dict = load_from_file(file_name_for_all_ner)
        all_stanford_text   = load_from_file(stanford_ner_pickle)
        for i,line in enumerate(f):
            print(i)
            line = line.split("\t")
            sen_num = line[0]
            route_to_root = word_to_route[sen_num]
            #sen = tokens_sentences[sen_num]
            #stanford = all_stanford_text[sen_num]
            #stanford = stanford_extract_ner_from_sen(sen)
            #all_stanford_text[sen_num] = stanford
            #stanford = all_stanford_text[sen_num]
            combine_processed_and_stanford = combine_two_sentences(stanford.copy(),processed_dict[sen_num])
            # combine_processed_and_stanford = processed_dict[sen_num]
            #combined_dict[sen_num] = combine_processed_and_stanford

            ners = extract_ner(combine_processed_and_stanford)
            ner_dict = check_person_and_location(ners)
            all_sentence_ner_dict[sen_num] = ner_dict

            text = sen_num + "\t"
            ner_dict =  all_sentence_ner_dict[line[0]]
            # if sen_num == 'sent1483':
            #     stanford = stanford_extract_ner_from_sen(tokens_sentences[sen_num])
            #     combine_processed_and_stanford = combine_two_sentences(stanford, processed_dict[sen_num])
            #     ners = extract_ner(combine_processed_and_stanford)
            #     ner_dict = check_person_and_location(ners)
            #     print(1)

            if not (person in ner_dict and location in ner_dict):
                continue


            possiable_persons, possiable_location = unique_person_and_location(ner_dict[person], ner_dict[location])
            for per in possiable_persons:
                person_appread_in = per[1]
                # loc = find_closest_location(ner_dict[location],person_appread_in)
                for loc in possiable_location:
                #     loc_appread_in = loc[1]
                    # if (len(ner_dict[person]) == 1 and len(ner_dict[location]) == 1):
                # if abs(loc_appread_in - person_appread_in) < 1500:
                    text_line = text + per[0] + "\tLive_In\t" + loc[0] + "\n"
                    save_all_text.append(text_line)

    write_to_file("naive_save_output.txt", save_all_text)
    #save_to_file(all_sentence_ner_dict, file_name_for_all_ner)

    #save_to_file(all_sentence_ner_dict, file_name_for_all_ner)
    # save_to_file(all_stanford_text,stanford_ner_pickle)
    #save_to_file(combined_dict,combind_sentences_pickle)
    evaluate_result.main("naive_save_output.txt","data/TRAIN.annotations")

if __name__ == '__main__':
   main()


