import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def logistic_reg_accuracy(z_splits: Tensor, target_splits: Tensor, train_keys=None, test_keys=None,
                          normalize=False, sample=1, validation=False):

    if train_keys is not None:
        x = torch.cat([z_splits[key] for key in train_keys])
        y = torch.cat([target_splits[key] for key in train_keys])
    else:
        x = z_splits
        y = target_splits

    if len(y.shape) > 2:
        raise ValueError('Target must be 1D or 2D (one hot vectors)')
    elif len(y.shape) == 2:
        y = y.argmax(-1)

    if normalize:
        x = (x - x.mean()) / x.std()

    if validation:
        clf = MLPClassifier(random_state=0, hidden_layer_sizes=[], activation='identity', solver='sgd', max_iter=1000,
                            early_stopping=True).fit(x[::sample], y[::sample])
        if test_keys is None:
            return clf.best_validation_score_

    else:
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(x[::sample], y[::sample])
        if test_keys is None:
            return clf.score(x, y)

    res = {}
    for key in test_keys:
        x = z_splits[key]
        y = target_splits[key]
        if len(y.shape) == 2:
            y = y.argmax(-1)
        res[key] = clf.score(x, y)
    return res


def replace_domain_with(z, y, c, domain_id, key):
    mask = c['train'] != domain_id
    train12_z = z['train'][mask]
    train12_y = y['train'][mask]
    train12_c = c['train'][mask]

    train12_val_z = torch.cat([train12_z, z[key]])
    train12_val_y = torch.cat([train12_y, y[key]])
    train12_val_c = torch.cat([train12_c, c[key]])

    return train12_val_z, train12_val_y, train12_val_c


def all_logistics(z, c, y, sample=1, validation=True):

    val_on_val = logistic_reg_accuracy(z, y, ['val'], sample=sample, validation=validation)

    train_val_z, train_val_y, train_val_c = replace_domain_with(z, y, c, 0, 'val')

    c_train = logistic_reg_accuracy(z, c, ['train'], sample=sample, validation=validation)
    c_val = logistic_reg_accuracy(train_val_z, train_val_c, sample=sample, validation=validation)

    c_perclass = {}
    train_val_y = train_val_y.argmax(-1)

    for c in range(y['train'].shape[-1]):
        train_val_z_class = train_val_z[train_val_y == c]
        train_val_c_class = train_val_c[train_val_y == c]
        c_perclass[c] = logistic_reg_accuracy(train_val_z_class, train_val_c_class,
                                              sample=sample, validation=validation)

    # for G2, split in-domain validation sets
    val_val_mask = torch.zeros(y['val'].shape[0])
    val_val_mask[::10] = 1

    z['val_val'] = z['val'][val_val_mask == 1]
    z['val'] = z['val'][val_val_mask == 0]
    y['val_val'] = y['val'][val_val_mask == 1]
    y['val'] = y['val'][val_val_mask == 0]

    trainval_on_all = logistic_reg_accuracy(z, y, ['train', 'val'],
                                            test_keys=['id_val', 'val_val'], sample=sample, validation=validation)

    result = {
        "trainval_on_train": trainval_on_all['id_val'],
        "trainval_on_val": trainval_on_all['val_val'],

        "val_on_val": val_on_val,
        "c_train": c_train,
        "c_val": c_val,
        "c_perclass": c_perclass,
    }

    return result


def all_logistics_test(z, c, y, sample=1, validation=True):

    val_on_val = logistic_reg_accuracy(z, y, ['val'], sample=sample, validation=validation)
    test_on_test = logistic_reg_accuracy(z, y, ['test'], sample=sample, validation=validation)

    train_val_z, train_val_y, train_val_c = replace_domain_with(z, y, c, 0, 'val')
    train_test_z, train_test_y, train_test_c = replace_domain_with(z, y, c, 0, 'test')

    c_train = logistic_reg_accuracy(z, c, ['train'], sample=sample, validation=validation)
    c_val = logistic_reg_accuracy(train_val_z, train_val_c, sample=sample, validation=validation)
    c_test = logistic_reg_accuracy(train_test_z, train_test_c, sample=sample, validation=validation)

    c_perclass = {}
    train_val_y = train_val_y.argmax(-1)

    c_perclass_test = {}
    train_test_y = train_test_y.argmax(-1)

    for c in range(y['train'].shape[-1]):
        train_val_z_class = train_val_z[train_val_y == c]
        train_val_c_class = train_val_c[train_val_y == c]
        c_perclass[c] = logistic_reg_accuracy(train_val_z_class, train_val_c_class,
                                              sample=sample, validation=validation)

        train_test_z_class = train_test_z[train_test_y == c]
        train_test_c_class = train_test_c[train_test_y == c]
        c_perclass_test[c] = logistic_reg_accuracy(train_test_z_class, train_test_c_class,
                                                   sample=sample, validation=validation)

    # for G2, split in-domain validation sets
    val_val_mask = torch.zeros(y['val'].shape[0])
    val_val_mask[::10] = 1

    z['val_val'] = z['val'][val_val_mask == 1]
    z['val'] = z['val'][val_val_mask == 0]
    y['val_val'] = y['val'][val_val_mask == 1]
    y['val'] = y['val'][val_val_mask == 0]

    trainval_on_all = logistic_reg_accuracy(z, y, ['train', 'val'],
                                            test_keys=['id_val', 'val_val'], sample=sample, validation=validation)

    test_val_mask = torch.zeros(y['test'].shape[0])
    test_val_mask[::10] = 1

    z['test_val'] = z['test'][test_val_mask == 1]
    z['test'] = z['test'][test_val_mask == 0]
    y['test_val'] = y['test'][test_val_mask == 1]
    y['test'] = y['test'][test_val_mask == 0]

    traintest_on_all = logistic_reg_accuracy(z, y, ['train', 'test'],
                                             test_keys=['id_val', 'test_val'], sample=sample, validation=validation)

    result = {
        "trainval_on_train": trainval_on_all['id_val'],
        "trainval_on_val": trainval_on_all['val_val'],

        "val_on_val": val_on_val,
        "c_train": c_train,
        "c_val": c_val,
        "c_perclass": c_perclass,

        "traintest_on_train": traintest_on_all['id_val'],
        "traintest_on_test": traintest_on_all['test_val'],

        "test_on_test": test_on_test,
        "c_test": c_test,
        "c_perclass_test": c_perclass_test
    }
    return result
