import torch
import torch.nn as nn
import dynet as dy
import numpy as np
# Device configuration
where_to_save_model = "files/save_mlp_model"

# Hyper-parameters
input_size = 30522
hidden_size_one = 1000
hidden_size_two = 100
output_size = 2
num_epochs = 600 * 4
batch_size = 100
learning_rate = 0.001
drop_out = 0.3



class OurNetwork(object):
    # The init method adds parameters to the parameter collection.
    def __init__(self, m):
        self.first_layer = m.add_parameters((hidden_size_one, 2*input_size))
        self.W = m.add_parameters((hidden_size_two, hidden_size_one))
        self.V = m.add_parameters((output_size, hidden_size_two))
        self.b = m.add_parameters((hidden_size_two))
        self.b_tag = m.add_parameters((output_size))
        dy.dropout(self.W, drop_out)
        dy.dropout(self.V, drop_out)
        dy.dropout(self.b, drop_out)
        dy.dropout(self.b_tag, drop_out)

    # the __call__ method applies the network to an input
    def __call__(self, inputs):
        first_layer = self.first_layer
        V = self.V
        W = self.W
        b = self.b
        b_tag = self.b_tag
        net_input = np.concatenate((inputs[0], inputs[1]))
        x = dy.inputTensor(net_input) # Row major
        after_one = dy.rectify(first_layer*x)
        net_output = dy.softmax(V * (dy.tanh(W * after_one) + b) + b_tag)
        return net_output

    def create_network_return_loss(self, inputs, expected_output):
        dy.renew_cg()
        out = self.__call__(inputs)
        loss = -dy.log(dy.pick(out, expected_output))
        # loss += self.V.value() * self.V.value()
        # loss += self.W.value() * self.W.value()
        # loss += self.b.value() * self.b.value()
        # loss += self.b_tag.value() * self.b_tag.value()
        return loss

    def create_network_return_best_and_loss(self, inputs,expected_output):
        dy.renew_cg()
        out = self(inputs)
        loss = -dy.log(dy.pick(out, expected_output))
        return np.argmax(out.npvalue()),loss.value()

    def return_best(self, inputs):
        dy.renew_cg()
        out = self(inputs)
        return np.argmax(out.npvalue())






def train_MLP(train,test):
    m = dy.ParameterCollection()
    network = OurNetwork(m)
    m.populate("files/save_mlp_model_0.328_1000")
    trainer = dy.SimpleSGDTrainer(m, learning_rate)
    # Train the model
    total_step = len(train)
    for epoch in range(num_epochs):
        cum_loss = 0.0
        for i, (label,vectors) in enumerate(train):
            # Move tensors to the configured device
            # labels = label.to(device)
            loss =network.create_network_return_loss(vectors,label)
            # Forward pass
            # Backward and optimize
            loss.value()  # need to run loss.value() for the forward prop
            if label:
                loss = 30 * loss

            loss.backward()
            trainer.update()
            cum_loss += loss.value()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, cum_loss ))

        print('Epoch Done [{}], AVG Loss: {:.4f}'
              .format(epoch + 1, cum_loss/len(train) ))
                # Test the model
                # In test phase, we don't need to compute gradients (for memory efficiency)
        correct = 0
        true_pos = 0
        true_neg = 0
        gold_true = 0
        gold_neg = 0
        false_pos = 0
        total = len(test)
        print("total test case = ", total)
        all_pred = []
        for i, (label, vectors) in enumerate(test):
            pred,loss = network.create_network_return_best_and_loss(vectors, label)
            correct += (pred == label)
            true_pos += (pred == label) and (label == 1)
            false_pos += (pred != label) and (label == 0)
            true_neg += (pred == label) and (label == 0)
            gold_true += (label == 1)
            gold_neg += (label == 0)
            all_pred.append(pred)
        f1 = 0
        recall = true_pos / gold_true
        prec = (true_pos+false_pos)
        if prec!=0:
            prec = true_pos / (true_pos+false_pos)
        if prec+recall!=0: f1 = 2 * prec * recall / (prec+recall)
        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
        print('gold true {} '.format(gold_true))
        print('we pred  {}  as 1'.format(sum(all_pred)))
        print('recall of the network on the test images: {} %'.format(100 * recall))
        print('prec of the network on the test images: {} %'.format(100 * prec))
        if (epoch == 0 or f1 > best_f1):
            print("Reached best F1 = ", f1)
            best_f1 = f1
            m.save(where_to_save_model + "_" + str(round(f1,3)) + "_"+ str(hidden_size_one))
    return all_pred

def grid_sreach():
    m = dy.ParameterCollection()
    network = OurNetwork(m)
    #m.populate("save_mlp_model_0.35406698564593303")
    trainer = dy.SimpleSGDTrainer(m, learning_rate)
    # Loss and optimizer


    # Save the model checkpoint