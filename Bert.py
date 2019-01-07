from pytorch_pretrained_bert import BertForMaskedLM, tokenization
import torch

model_name = 'bert-large-uncased'
bert = BertForMaskedLM.from_pretrained(model_name)
tokenizer = tokenization.BertTokenizer.from_pretrained(model_name)
bert.eval()


def get_predictions(sent):
    pre, target, post = sent.split('***')
    if 'mask' in target.lower():
        target = ['[MASK]']
    else:
        target = tokenizer.tokenize(target)
    tokens = ['[CLS]'] + tokenizer.tokenize(pre)
    # print(tokens)

    target_idx = len(tokens)
    # print("target_idx = ", target_idx)
    tokens += target + tokenizer.tokenize(post) + ['[SEP]']
    # print("tokens = ", tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(len(input_ids))
    tens = torch.LongTensor(input_ids).unsqueeze(0)
    # print(tens)
    res = bert(tens)[0, target_idx]
    return res.data.numpy()
    # print("res before softmax = ", res.data.numpy())
    #
    # res = torch.nn.functional.softmax(res, -1)
    # print("res after softmax = ", res.shape)
    # probs, best_k = torch.topk(res, 50)
    # best_k = [int(x) for x in best_k]
    # probs = [float(x) for x in probs]
    # best_k = tokenizer.convert_ids_to_tokens(best_k)
    # return list(zip(best_k, probs))


#
# for w,p in get_predictions(                                   ################
#     ''' Once upon a time there was a very rich man who loved wines and lived with his three daughters.
#         The two older daughters laughed at anyone who did not dress as well as they did.
#         If the two of them were not resting at home, they were out shopping for as many  ***mask*** as they could carry home.
#      '''):
#     print(f'{w}({100*p:.2f}%)')
#


sentence = [
    "Israel television rejected a skit by comedian Tuvia Tzafir that attacked public apathy by depicting an Israeli family watching TV while a fire raged ***mask*** .", ''' Israel television rejected a skit by comedian ***mask*** that attacked 
        public apathy by depicting an Israeli family watching TV while a fire raged outside .''']


def get_vectors_from_bert(list_of_sentences):
    person_vector = get_predictions(list_of_sentences[0])
    location_vector = get_predictions(list_of_sentences[1])
    return person_vector, location_vector
