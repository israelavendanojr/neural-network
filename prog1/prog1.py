import parser
from layer import Layer
from dataset import DataSet
from data_loader import DataLoader
import numpy as np

args = vars(parser.parse_arguments())

def create_network():
    hidden_units = args['nunits']
    num_layers = args['nlayers']
    output_dim = args['output_dim']

    dataset = DataSet(args)
    network = [Layer] * num_layers
    
    # instantiate 
    d = len(dataset.data_train[0].feat)
    network[0] = Layer(d, hidden_units, args)
    
    # instantiate layers 2 through l-1 
    for i in range(1, len(network) - 1):
        network[i] = Layer(hidden_units, hidden_units, args)

    if num_layers > 1:
        network[-1] = Layer(hidden_units, output_dim, args)

    return network

def forward_pass(network, mini_batch_feat):
    hidden_act = args['hidden_act']
    # first
    
    network[0].z = network[0].w.T @ mini_batch_feat + network[0].b

    network[0].a = apply_activation(hidden_act, network[0].z)
    
    # middle
    for i in range(1, len(network) - 1):
        current = network[i]
        prev = network[i-1]

        current.z = current.w.T @ prev.a + current.b
        current.a = apply_activation(hidden_act, current.z)

    # last
    if len(network) > 1:
        current = network[-1]
        prev = network[-2]
        current.z = current.w.T @ prev.a + current.b
        current.a = output_activation(current.z)
    
def backward_pass(network, mini_batch_target, mini_batch_feat):
    hidden_act = args['hidden_act']
    type = args['type']
    output_dim = args['output_dim']

    one_vector = np.ones(len(mini_batch_target[0]))
    one_vector = one_vector[:, None].T

    # last
    if type == 'C' and output_dim > 1:
        one_hot = np.zeros_like(network[-1].a.T)
        
        for i in range(len(one_hot)):
            j = int(mini_batch_target[0][i])
            one_hot[i][j] = 1
            
        network[-1].delta = network[-1].a - one_hot.T
    else:
        network[-1].delta = network[-1].a - mini_batch_target

    network[-1].b_grad = (one_vector @ network[-1].delta.T) / len(mini_batch_target[0])
    network[-1].w_grad = (network[len(network) - 2].a @ network[-1].delta.T) / len(mini_batch_target[0])

    # middle
    for i in range(len(network) -2, 0, -1):
        f_prime = act_derivative(hidden_act, network[i].z)
        matrix = network[i+1].w @ network[i+1].delta
        network[i].delta = np.multiply(f_prime, matrix)

        network[i].b_grad =  (one_vector @ network[i].delta.T) / len(mini_batch_target[0])
        network[i].w_grad = (network[i-1].a @ network[i].delta.T) / len(mini_batch_target[0])

    
    if (len(network) > 1):
        # first
        f_prime = act_derivative(hidden_act, network[0].z)
        matrix = network[1].w @ network[1].delta
        network[0].delta = np.multiply(f_prime, matrix)

        network[0].b_grad = (one_vector @ network[0].delta.T) / len(mini_batch_target[0])
        network[0].w_grad = (mini_batch_feat @ network[0].delta.T) / len(mini_batch_target[0])

def apply_activation(act, z):
    a = np.zeros(z.shape)
    
    if act == 'sig':
        sig = lambda x: 1 / (1 + np.e ** -x)
        
        a = sig(z)
    elif act == 'tanh':
        tanh = lambda x: 2 * ( 1 / (1 + np.e ** -x)) - 1
        a = tanh(z)
    else:
        a = np.maximum(z, np.zeros_like(z))
        
    return a

def output_activation(z):
    type = args['type']
    output_dim = args['output_dim']
    a = np.zeros(z.shape)

    if type == 'R':
        a = z
    elif type == 'C':
        if output_dim > 2:
            exp = (np.e ** z)
            sum = np.sum(exp, 0)
            sum = sum[None, :]
            sum = np.tile(sum, (output_dim, 1))
            a = exp / sum
        else:
            sig = lambda x: 1 / (1 + np.e ** -x)
            a = sig(z)
    
    return a

def act_derivative(act, z):
    a = np.zeros(z.shape)

    if act == 'sig':
        sig = lambda x: apply_activation(act, z) * (1 - apply_activation(act, z))
        a = sig(z)
    elif act == 'tanh':
        tanh = lambda x: 1 - apply_activation(act, z) ** 2
        a = tanh(z)
    else:
        for i in range(len(a)):
            for j in range(len(a[0])):
                if z[i][j] < 0:
                    a[i][j] = 0
                else:
                    a[i][j] = 1

    return a

def single_pass(network, data_loader):
    # get minibatch
    mb = args['mb']
    if mb == 0 or mb > len(data_loader.train_set):
        batch = data_loader.train_set
        data_loader.updates = data_loader.updates + 1
        data_loader.epoch = data_loader.epoch + 1
    else:
        batch = data_loader.get_batch()
    # decouple batch
    feat_batch = []
    target_batch = []
    for i in range(len(batch)):
        feat_batch.append(batch[i].feat)
        target_batch.append(batch[i].target)


    feat_batch = np.array(feat_batch).T
    target_batch = np.array(target_batch).T
    # passes
    forward_pass(network, feat_batch)
    backward_pass(network, target_batch, feat_batch)

    # updating learn params
    learning_rate = args['learnrate']
    
    for i in range(len(network)):
        network[i].w = network[i].w - learning_rate * network[i].w_grad
        network[i].b = network[i].b - learning_rate * network[i].b_grad.T

    # report stuff
    report_freq = args['report_freq']
    if data_loader.updates % report_freq == 0:
        
        z = []
        a = []
        y = []
    
        act = args['hidden_act']
        # strips data halves
        for i in range(len(data_loader.dev_set)):
            a.append(data_loader.dev_set[i].feat)
            y.append(data_loader.dev_set[i].target)

        a = np.array(a).T
        y = np.array(y).T

        for i in range(len(network) - 1):
            z = network[i].w.T @ a + network[i].b
            a = apply_activation(act, z)
        
        z = network[-1].w.T @ a + network[-1].b
        a = output_activation(z)


        type = args['type']
        verbose = args['v']
        mb_string = ''

        # if regression, else classification
        if type == 'R':
            diff = lambda x, y: (x-y)**2
            dev_sqr_err = np.sum(diff(a,y)) / len(y[0])
            

            if verbose:
                pred = network[-1].a
                actual = target_batch
                mb_sqr_err = np.sum(diff(pred,actual)) / len(actual[0])
                mb_string = ' minibatch=' + '%.3f'%(mb_sqr_err)
                
            
            print('Epoch ', '%04d'%(data_loader.epoch), "UPDATE ", '%06d'%(data_loader.updates), ":", mb_string, " dev=", '%.3f'%dev_sqr_err)

        else:
            count = 0
            output_dim = args['output_dim']
            # if multiclass, else binary
            if output_dim > 1:
                a = a.T
                y = y.T
                for i in range(len(a)):
                    pred_class = np.argmax(a[i])
                    if pred_class == y[i]:
                        count = count + 1
                
                dev_accuracy = count / len(y)
            else:
                for i in range(len(y)):
                    for j in range(len(y[0])):
                        if y[i][j] == 1:
                            if a[i][j] >= .5:
                                count = count + 1
                        else:
                            if a[i][j] < .5:
                                count = count + 1
            
                dev_accuracy = count / len(y[0])

            # report minibatch in verbose mode
            if verbose:
                pred = network[-1].a
                actual = target_batch
                mb_count = 0

                if output_dim > 1:
                    pred = pred.T
                    actual = actual.T
                    for i in range(len(pred)):
                        pred_class = np.argmax(pred[i])
                        if pred_class == actual[i]:
                            mb_count = mb_count + 1
                    
                    mb_accuracy = mb_count / len(actual)
                    mb_string = ' minibatch=' + '%.3f'%(mb_accuracy)

                else:
                    for i in range(len(actual)):
                        for j in range(len(actual[0])):
                            if actual[i][j] == 1:
                                if pred[i][j] >= .5:
                                    mb_count = mb_count + 1
                            else:
                                if pred[i][j] < .5:
                                    mb_count = mb_count + 1
                
                    mb_accuracy = mb_count / len(actual[0])
                    mb_string = ' minibatch=' + '%.3f'%(mb_accuracy)

            print('Epoch ', '%04d'%(data_loader.epoch), "UPDATE ", '%06d'%(data_loader.updates), ":", mb_string, " dev=", '%.3f'%dev_accuracy)

def main():
    dataset = DataSet(args)
    data_loader = DataLoader(dataset.data_train, dataset.data_dev, args['mb'])
    network = create_network()

    total_updates = args['total_updates']
    for i in range(total_updates):
        single_pass(network, data_loader)
    

if __name__ == '__main__':
    main()
