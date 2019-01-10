import numpy as np
import sys
from collections import Counter
count = Counter()
all_tags = {}
feature_map = {}
count_feature = {}
not_rare_words = set()
from utils import write_to_file


def create_dicts(input_file_name):
    file_read = open(input_file_name)
    for line_number, line in enumerate(file_read):
        features = line.strip().split(" ")
        for featur in features[1:]:
            if not featur in feature_map: # and count_feature[featur] > 9:
                feature_map[featur] = (len(feature_map))


def list_of_index(line,feature_dict):
    feature_index_per_word = []
    features = line.strip().split(" ")
    string_of_line = str(features[0])
    for f in features[1:]:
        if f in feature_dict:
            feature_index_per_word.append(feature_dict[f])

    feature_index_per_word = sorted(feature_index_per_word)
    return string_of_line, feature_index_per_word


def line_to_feature_str(line,feature_dict):
    string_of_line, feature_index_per_word = list_of_index(line,feature_dict)
    assert (len(feature_index_per_word) == len(set(feature_index_per_word)))

    for f_str in feature_index_per_word:
        string_of_line += " " + str(f_str) + ":1"

    string_of_line += "\n"

    return string_of_line


def generate_vector(input_file_name):

    f = open(input_file_name)
    all_lines = []

    for line_number, line in enumerate(f):
        all_lines.append(line_to_feature_str(line,feature_map))


    all_lines[-1] = all_lines[-1][:-1]
    return all_lines


def generate_tags_for_feature(pos, index):
    tag = pos[index]
    if index > 2:
        prev_tag = pos[index - 1]
        prev_prev_tag = pos[index-2]
    elif index == 1:
        prev_tag = pos[index - 1]
        prev_prev_tag = None
    else:
        prev_tag = None
        prev_prev_tag = None

    return tag,prev_tag,prev_prev_tag


def append_set_to_file(file_name,s,command = "a"):
    text = []
    string = ""
    for nrw in s:
        string += nrw + " "

    string = string[:-1]
    text.append(string)

    write_to_file(file_name,text)


def write_dict_to_file(file_name, d , command = "w"):
        f = open(file_name, command)
        text = []
        for i,key in enumerate(d):
             string =  str(key) + " "+ str(d[key]) + "\n"
             if d[key] == 1:
                 print
             text.append(string)


        write_to_file(file_name,text)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def main(input_file_name= "memm-features",features_vec_file="vec_file.txt",features_map_file="feature_map_file.txt"):
    create_dicts(input_file_name)
    text = generate_vector(input_file_name)


    write_to_file(features_vec_file, text)
    # write_dict_to_file(features_map_file, all_tags )
    write_dict_to_file(features_map_file,feature_map)
    return features_vec_file,features_map_file

if __name__ == '__main__':
    main()