
def remove_dot(st):
    if st[-1] == '.':
        return st[:-1]
    # if st.lower().startswith("ms. "):
    #     return st[4:]
    # if st.lower().startswith("ms "):
    #     return st[3:]
    # if st.lower().startswith("msr "):
    #     return st[4:]
    # if st.lower().startswith("msr. "):
    #     return st[5:]
    return st

def gold_file(gold_file_name = "data/TRAIN.annotations"):
    from utils import relation
    sentnce_to_relation = {}
    gold_items = set()
    with open(gold_file_name) as f:
        for line in f:
            if relation in line:
                line = line.split("\t")
                sen_num = line[0]
                per = line[1]
                loc = line[3]
                per = remove_dot(per)
                loc = remove_dot(loc)
                sentnce_to_relation[sen_num] = sentnce_to_relation.get(sen_num,list())
                sentnce_to_relation[sen_num].append((per,loc))
                if (sen_num, per, loc) in gold_items:
                    print("duplicate ",(sen_num, per, loc))
                gold_items.add((sen_num, per, loc))

    return sentnce_to_relation,gold_items


def read_pred(pred_file_name,sentnce_to_relation):
    pred_set = set()
    good = 0.0
    bad = 0.0
    i = 0
    with open(pred_file_name,"r") as f:
        for line in f:
            i+=1
            line = line.strip().split("\t")
            per = line[1]
            loc = line[3]
            per = remove_dot(per)
            loc = remove_dot(loc)
            sen_num = line[0]
            if ((sen_num,per,loc)) in pred_set:
                print((sen_num,per,loc))
                print("pred twice same value")
                exit()
            pred_set.add((sen_num,per,loc))

            if sen_num in sentnce_to_relation:
                pair = set(sentnce_to_relation[sen_num])
                if (per,loc) in pair:
                    good += 1
                else:
                    bad += 1
                    print(sen_num,per,loc)
            else:
                bad+=1
                print(sen_num, per, loc)

    return good,bad,pred_set

def main(pred_file_name = "save_output.txt",golden_file_name ="data/DEV.annotations" ):
    # gold_file_name = "data/TRAIN.annotations"
    sentnce_to_relation, gold_items = gold_file(golden_file_name)
    good, bad, pred_set = read_pred(pred_file_name,sentnce_to_relation)


    print()
    print()
    print ("missed relations")
    for mis_ele in (gold_items - pred_set):
        print( mis_ele  )

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print ("good=",good)
    print ("bad=",bad)

    if good + bad > 0: prec = good / (good + bad)
    else: prec = 0
    recall = 1 - len(gold_items - pred_set) / len(gold_items)


    print("len of all Live in ", len(gold_items))

    print ("prec " + str(prec))
    print ("recall " + str(recall))
    print("F1 score ", 2*prec*recall/(prec+recall))
    return (gold_items - pred_set)

if __name__ == '__main__':
   main()


