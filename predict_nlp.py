from allennlp.predictors import Predictor
dependency_parser = False
ner_tagger = True
# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz") # NER doesnt work
# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")  #Constituency Parsing
# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz") # Dependency parsing


if (dependency_parser):
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz") # Dependency parsing
elif (ner_tagger):
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model elmo-2018.08.31.tar.gz ner-examples.jsonl")
else:
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")  # Constituency Parsing

results = predictor.predict(sentence="""Dominique Beuff of the International Committee of the Red Cross in Khartoum
                                        said he had reports of people dying at the rate of up to four a day in some areas of the south affected by a 5-year-old civil war .""")

for cat in results:
    print ("cat ", cat)
    print (results[cat])

print (len(results["words"]))
print (len(results["predicted_dependencies"]))
print (len(results["predicted_heads"]))
if (dependency_parser):
    for word, tag , head,POS in zip(results["words"], results["predicted_dependencies"],results["predicted_heads"],results["pos"]):
        print(f"{word}\t{POS}\t{tag}\t{results['words'][head]}")
else:
    for word, tag in zip(results["tokens"], results["pos_tags"]):
        print(f"{word}\t{tag}")