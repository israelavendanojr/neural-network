class DataSet:

    def __init__(self, args):
        data_x_train = self.file_reader(args['train_feat'])
        data_y_train = self.file_reader(args['train_target'])
        self.data_train = []

        for i in range(len(data_x_train)):
            data = Data(data_x_train[i], data_y_train[i])
            self.data_train.append(data)

        data_x_dev = self.file_reader(args['dev_feat'])
        data_y_dev = self.file_reader(args['dev_target'])

        self.data_dev = []

        for i in range(len(data_x_dev)):
            data = Data(data_x_dev[i], data_y_dev[i])
            self.data_dev.append(data)
            

    def file_reader(self,filename):
        # open file for reading
        file = open(filename)
        
        # define 2d array
        data = []

        # read line by line, and store data in array
        for line in file:
            data_point = line.strip().split(' ')

            # typecast from char to int
            for i in range(len(data_point)):
                data_point[i] = float(data_point[i])
                
            data.append(data_point)
        
        return data

class Data:

    def __init__(self, feat, target):
        self.feat = feat
        self.target = target