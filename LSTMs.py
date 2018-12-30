import dynet as dy
import numpy as np


class Bi_LSTM_word_embedding(object):
    def __init__(self,model,nwords,ntags):
        self.E = model.add_lookup_parameters((nwords, 128))
        self.pO = model.add_parameters((ntags, 50 * 2))

        self.first_layer_builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model),
        ]
        self.second_layer_builders = [
            dy.LSTMBuilder(1, 2 * 50, 50, model),
            dy.LSTMBuilder(1, 2 * 50, 50, model),
        ]


    def build_tagging_graph(self,words, tags):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.first_layer_builders]

        wembs = [self.E[w] for w in words]
        wembs = [dy.noise(we, 0.1) for we in wembs]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        errs = []
        output_from_first_layer = [dy.concatenate([f, b]) for f,b in zip(fw,reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer_builders]
        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = self.pO * f_b

            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self,sent,vt,vw,UNK):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.first_layer_builders]
        wembs = [self.E[vw.w2i.get(w, UNK)] for w, t in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        tags = []
        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer_builders]
        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        for f, b, (w, t) in zip(fw, reversed(bw), sent):
            r_t = self.pO * dy.concatenate([f, b])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags

    def predict_tags(self,sent,vt,vw,UNK):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.first_layer_builders]
        wembs = [self.E[vw.w2i.get(w[0], UNK)] for w in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer_builders]
        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        tags = []
        for f, b, w in zip(fw, reversed(bw), sent):
            r_t = self.pO * dy.concatenate([f, b])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags



class Bi_LSTM_char_embedding(object):
    def __init__(self,model,nchar,ntags):
        self.INPUT_DIM = 128
        self.output_from_char_lstm = 50
        self.E = model.add_lookup_parameters((nchar, self.INPUT_DIM))
        self.pO = model.add_parameters((ntags, 50 * 2))

        self.first_layer = [
            dy.LSTMBuilder(1, self.output_from_char_lstm, 50, model),
            dy.LSTMBuilder(1, self.output_from_char_lstm, 50, model),
        ]
        self.second_layer = [
            dy.LSTMBuilder(1, 50*2, 50, model),
            dy.LSTMBuilder(1, 50*2, 50, model),
        ]
        self.char_builder =dy.LSTMBuilder(1, self.INPUT_DIM,self.output_from_char_lstm, model)

    def convert_words_to_vecs(self,words):
        word_rep = []
        for w in words:
            char_init = self.char_builder.initial_state()
            input_vecs = [self.E[c] for c in w]
            final_state = char_init.transduce(input_vecs)
            word_rep.append(final_state[-1])
        return word_rep

    def build_tagging_graph(self,words, tags):
        dy.renew_cg()
        #self.lstm = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        wembs = self.convert_words_to_vecs(words)
        wembs = [dy.noise(we, 0.1) for we in wembs]

        f_init, b_init = [b.initial_state() for b in self.first_layer]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        errs = []

        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = self.pO * f_b

            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self,sent,vt,vw,UNK):
        dy.renew_cg()
        wembs = self.convert_words_to_vecs(sent)

        f_init, b_init = [b.initial_state() for b in self.first_layer]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]
        output_from_first_layer = [dy.concatenate([f,b]) for f,b in zip(fw,reversed(bw)) ]

        f_init, b_init = [b.initial_state() for b in self.second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        tags = []
        for f, b in zip(fw, reversed(bw)):
            r_t = self.pO * dy.concatenate([f, b])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags


class Bi_LSTM_SUBWORDS_embedding(object):
    def __init__(self,model,nwords,ntags,preffix_size,suffix_size):
        self.E = model.add_lookup_parameters((nwords, 128))
        self.preffix = model.add_lookup_parameters((preffix_size, 128))
        self.suffix = model.add_lookup_parameters((suffix_size, 128))
        self.pO = model.add_parameters((ntags, 50 * 2))

        self.first_layer = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model),
        ]
        self.second_layer = [
            dy.LSTMBuilder(1, 2*50, 50, model),
            dy.LSTMBuilder(1, 2*50, 50, model),
        ]


    def build_tagging_graph(self, words, tags):
        dy.renew_cg()
        prefix_indices = [p[0] for p in words]
        suffix_indices = [p[2] for p in words]
        words = [p[1] for p in words]
        wembs = []
        for w, p, s in zip(words, prefix_indices, suffix_indices):
            we = self.E[w]
            pe = self.preffix[p]
            se = self.suffix[s]
            wembs.append(dy.esum([we,pe,se]))

        f_init, b_init = [b.initial_state() for b in self.first_layer]

        #wembs = [self.Word_E[w] for w in words]
        wembs = [dy.noise(we, 0.1) for we in wembs]



        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        errs = []

        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = self.pO * f_b

            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self,sent,vt,vw,UNK):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.first_layer]
        #wembs = [self.Word_E[vw.w2i.get(w, UNK)] for w, t in sent]
        wembs = []
        for p,w,s in sent:
            we = self.E[w]
            pe = self.preffix[p]
            se = self.suffix[s]
            wembs.append(dy.esum([we,pe,se]))

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        tags = []
        for f, b, (w, t,s) in zip(fw, reversed(bw), sent):
            r_t = self.pO * dy.concatenate([f, b])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags


class Bi_LSTM_W_AND_C_embedding(object):
    def __init__(self,model, nwords, nchars ,ntags):
        self.Word_E = model.add_lookup_parameters((nwords, 128))
        self.char_E = model.add_lookup_parameters((nchars, 128))
        self.pO = model.add_parameters((ntags, 50 * 2 * 2))

        self.word_first_layer = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model),
        ]

        self.word_second_layer = [
            dy.LSTMBuilder(1, 2*50, 50, model),
            dy.LSTMBuilder(1, 2*50, 50, model),
        ]

        self.char_flow_second_layer = [
            dy.LSTMBuilder(1, 2 * 50, 50, model),
            dy.LSTMBuilder(1, 2 * 50, 50, model),
        ]

        self.char_flow_first_layer = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model),
        ]
        self.char_builder = dy.LSTMBuilder(1, 128 , 128, model)
    #####

    def convert_words_to_vecs(self,words):
        word_rep = []
        for w in words:
            char_init = self.char_builder.initial_state()
            input_vecs = [self.char_E[c] for c in w]
            final_state = char_init.transduce(input_vecs)
            word_rep.append(final_state[-1])
        return word_rep

    def build_tagging_graph_for_chars(self,words):
        #self.lstm = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        wembs = self.convert_words_to_vecs(words)
        #wembs = [self.E[w] for w in words]
        wembs = [dy.noise(we, 0.1) for we in wembs]

        f_init, b_init = [b.initial_state() for b in self.char_flow_first_layer]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.char_flow_second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        vector_result = []

        for f, b in zip(fw, reversed(bw)):
            f_b = dy.concatenate([f, b])
            vector_result.append(f_b)

        return vector_result
    #####
    def build_tagging_graph(self,words, tags):
        dy.renew_cg()
        words_for_char = [w[1] for w in words]
        words = [w[0] for w in words]
        f_init, b_init = [b.initial_state() for b in self.word_first_layer]

        wembs = [self.Word_E[w] for w in words]
        wembs = [dy.noise(we, 0.1) for we in wembs]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.word_second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        errs = []
        char_lstm_vectors = self.build_tagging_graph_for_chars(words_for_char)

        for f, b, chars_vec, t in zip(fw, reversed(bw), char_lstm_vectors ,tags):
            f_b = dy.concatenate([f, b])
            con_cat = dy.concatenate([f_b,chars_vec])
            r_t = self.pO * con_cat

            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self,sent,vt,vw, UNK, words_for_char):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.word_first_layer]
        wembs = [self.Word_E[vw.w2i.get(w, UNK)] for w, t in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.word_second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        char_lstm_vectors = self.build_tagging_graph_for_chars(words_for_char)

        tags = []
        for f, b, char_vec,(w, t) in zip(fw, reversed(bw), char_lstm_vectors ,sent):
            r_t = self.pO * dy.concatenate([f, b,char_vec])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags


    def predict_tags(self,sent,vt,vw, UNK, words_for_char):
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.word_first_layer]
        wembs = [self.Word_E[vw.w2i.get(w[0], UNK)] for w in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        output_from_first_layer = [dy.concatenate([f, b]) for f, b in zip(fw, reversed(bw))]

        f_init, b_init = [b.initial_state() for b in self.word_second_layer]

        fw = [x.output() for x in f_init.add_inputs(output_from_first_layer)]
        bw = [x.output() for x in b_init.add_inputs(reversed(output_from_first_layer))]

        char_lstm_vectors = self.build_tagging_graph_for_chars(words_for_char)
        tags = []
        for f, b, char_vec in zip(fw, reversed(bw), char_lstm_vectors):
            r_t = self.pO * dy.concatenate([f, b,char_vec])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(vt.i2w[chosen])
        return tags