import numpy as np

'''return: shuffled data with indice'''
def to_categorical(labels, num_classes):
    category_label = []
    for label in labels:
        vec = [0]*num_classes
        for l in label:
            vec[l] = 1
        category_label.append(vec)
    return category_label

def paired_shuffle(dats, labels):
    indice = [idx for idx in range(len(dats))]

    np.random.shuffle(indice)
    shuffled_dats = []
    shuffled_labels = []
    for idx in indice:
        shuffled_dats.append(dats[idx])
        shuffled_labels.append(labels[idx])
    return (shuffled_dats, shuffled_labels, indice)

def index_shuffle(data_len):
    return np.random.permutation(data_len)

def queue_sort(queue):
    input_ex =[]
    while True:
        try:
            (i, dat) =queue.get_nowait()
            input_ex.append((i, dat))
        except:
            break
    input_ex =sorted(input_ex, key=lambda x: x[0])
    return [item[1] for item in input_ex]


def kfold(n_sample, n_split=5, shuffle=False):
    indice = [idx for idx in range(n_sample)]
    if shuffle:
        np.random.shuffle(indice)

    split_size = n_sample // n_split
    for ns in range(n_split):
        if (ns+1)*split_size > n_sample:
            idx_val = indice[ns*split_size:]
        else:
            idx_val = indice[ns*split_size: (ns+1)*split_size]
        idx_train = [idx for idx in filter(lambda x: x not in idx_val, indice)]

        yield idx_train, idx_val

# balanced data split per classes using sample-and-reject method
def data_split_balance(labels, num_classes, ratio=[5,1], shuffle=False):
    n_sample = len(labels)
    label_index = [idx for idx in range(n_sample)]
    if shuffle:
        np.random.shuffle(label_index)

    label_num_train = [0]*num_classes
    label_num_val = [0]*num_classes

    label_num = [0]*num_classes

    # calculate label statistics
    for label in labels:
        for l in label:
            label_num[l] += 1

    label_num_sorted, label_class_idx = sort_index(label_num)

    label_train = []
    label_val = []

    for idx_class, cls in enumerate(label_class_idx):
        #allocate each class from the fewest labels

        num_train = label_num_sorted[idx_class]*ratio[0]//(ratio[0]+ratio[1])
        # num_val = vals[idx] - num_train

        for lidx in label_index:
            label = labels[lidx]
            for l in label:
                if l == cls:
                    if label_num_train[l] < num_train and not lidx in label_train and not lidx in label_val:
                        # if training set is not filled
                        label_num_train[l] += 1
                        for l_other in label:
                            if not l_other == cls:
                                label_num_train[l_other] += 1
                        label_train.append(lidx)

                    else:
                        if not lidx in label_val and not lidx in label_train:
                            # fill the rest set to the val data set
                            label_num_val[l] += 1
                            for l_other in label:
                                if not l_other == cls:
                                    label_num_val[l_other] += 1
                            label_val.append(lidx)
        a = 1


    return label_train, label_val, label_num_train, label_num_val


def sort_index(vals):
    comp = [z for z in zip(vals, range(len(vals)))]
    sort_comp = sorted(comp, key=lambda x: x[0])

    return [s[0] for s in sort_comp], [s[1] for s in sort_comp]




