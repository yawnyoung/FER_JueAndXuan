from seq_fer_datasets import *
from seq_fer import SFER_LSTM
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import time
from matplotlib.gridspec import GridSpec
import math
import random
import collections
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def save_checkpoint(state_dict, is_best, filename):
    if is_best:
        print('=> Saving a new best')
        torch.save(state_dict, filename)
    else:
        print('=> Validation accuracy did not improve')


def eval_results(nn_model, data_loader, disp=False):
    """
    Evaluate results of a trained model on a dataset
    :param nn_model: the trained model
    :param data_loader: the dataset loader
    :return:
    """
    global expr_classes

    nn_model.eval()

    num_true = 0
    num_data = 0
    for sample_batched in data_loader:
        output, sorted_idx = nn_model(sample_batched)
        max_values, predicted = torch.max(output.data, 1)
        for pred, idx in zip(predicted, sorted_idx):
            if pred == sample_batched[1][idx]:
                num_true += 1
            num_data += 1
        if disp:
            print('{:16}{}'.format('ground truth: ', ['{:8}'.format(expr_classes[sample_batched[1][idx]]) for idx in sorted_idx]))
            print('{:16}{}'.format('predict: ', ['{:8}'.format(expr_classes[pred]) for pred in predicted]))

    # accuracy
    accuracy = float(num_true) / float(num_data)

    return accuracy


def get_expr_label(label_path):
    label_files = glob.glob(os.path.join(label_path, '*.txt'))

    em = 0
    with open(label_files[0]) as f:
        for line in f:
            line = line.lstrip(' ')
            line = line.split('.')
            em = int(line[0])

    return em


def split_datset_same_dist(data_paths, label_paths, train_ratio=0.6):

    global train_file, valid_file, test_file

    # re-arrange data and label according to the original label distribution
    arranged_data_paths = [[] for _ in range(8)]
    arranged_label_paths = [[] for _ in range(8)]

    for data_path, label_path in zip(data_paths, label_paths):
        expr_idx = get_expr_label(label_path)
        arranged_data_paths[expr_idx].append(data_path)
        arranged_label_paths[expr_idx].append(label_path)

    train_data_paths = []
    train_label_paths = []

    valid_data_paths = []
    valid_label_paths = []

    test_data_paths = []
    test_label_paths = []

    good_split = False

    while not good_split:
        for sgl_class_data_paths, sgl_class_label_paths in zip(arranged_data_paths, arranged_label_paths):
            all_size = len(sgl_class_data_paths)

            indices = list(range(all_size))
            random.shuffle(indices)

            test_ratio = (1 - train_ratio) / 2

            test_size = math.floor(all_size * test_ratio)
            valid_size = test_size

            if sgl_class_data_paths:

                test_data_paths += [sgl_class_data_paths[idx] for idx in indices[:test_size]]
                test_label_paths += [sgl_class_label_paths[idx] for idx in indices[:test_size]]

                valid_data_paths += [sgl_class_data_paths[idx] for idx in indices[test_size:test_size + valid_size]]
                valid_label_paths += [sgl_class_label_paths[idx] for idx in indices[test_size:test_size + valid_size]]

                train_data_paths += [sgl_class_data_paths[idx] for idx in indices[test_size + valid_size:all_size]]
                train_label_paths += [sgl_class_label_paths[idx] for idx in indices[test_size + valid_size:all_size]]

        # count subjects
        train_subjects = [data.split('/')[6] for data in train_data_paths]
        valid_subjects = [data.split('/')[6] for data in valid_data_paths]
        test_subjects = [data.split('/')[6] for data in test_data_paths]

        v_not_included = [v_s for v_s in valid_subjects if v_s not in train_subjects]
        t_not_included = [t_s for t_s in test_subjects if t_s not in train_subjects]
        print(v_not_included)
        print(t_not_included)

        if not v_not_included and not t_not_included:
            good_split = True
        else:
            del train_data_paths[:]
            del train_label_paths[:]
            del valid_data_paths[:]
            del valid_label_paths[:]
            del test_data_paths[:]
            del test_label_paths[:]

    # store train data paths
    with open(train_file, mode='w') as tf:
        for data in train_data_paths:
            path_list = data.split('/')
            tf.write(os.path.join(path_list[-2], path_list[-1])+'\n')

    with open(test_file, mode='w') as tf:
        for data in test_data_paths:
            path_list = data.split('/')
            tf.write(os.path.join(path_list[-2], path_list[-1]) + '\n')

    with open(valid_file, mode='w') as tf:
        for data in valid_data_paths:
            path_list = data.split('/')
            tf.write(os.path.join(path_list[-2], path_list[-1]) + '\n')


def gen_vl_paths(train_file, valid_file, test_file):

    global video_root_dir, label_root_dir

    train_v_paths = []
    train_l_paths = []
    valid_v_paths = []
    valid_l_paths = []
    test_v_paths = []
    test_l_paths = []

    with open(train_file) as f:
        for line in f:
            train_v_paths.append(os.path.join(video_root_dir, line.rstrip('\n')))
            train_l_paths.append(os.path.join(label_root_dir, line.rstrip('\n')))

    with open(valid_file) as f:
        for line in f:
            valid_v_paths.append(os.path.join(video_root_dir, line.rstrip('\n')))
            valid_l_paths.append(os.path.join(label_root_dir, line.rstrip('\n')))

    with open(test_file) as f:
        for line in f:
            test_v_paths.append(os.path.join(video_root_dir, line.rstrip('\n')))
            test_l_paths.append(os.path.join(label_root_dir, line.rstrip('\n')))

    return train_v_paths, train_l_paths, valid_v_paths, valid_l_paths, test_v_paths, test_l_paths


def plot_dataset_distribution(train_l_paths, valid_l_paths, test_l_paths):

    # check expression distribution in train, validation and test sets
    train_label_dist = [0] * 8
    valid_label_dist = [0] * 8
    test_label_dist = [0] * 8

    for label_path in train_l_paths:
        expr = get_expr_label(label_path)
        train_label_dist[expr] += 1

    for label_path in valid_l_paths:
        expr = get_expr_label(label_path)
        valid_label_dist[expr] += 1

    for label_path in test_l_paths:
        expr = get_expr_label(label_path)
        test_label_dist[expr] += 1

    the_grid = GridSpec(1, 3, wspace=0.5)

    the_grid.update(left=0.05, right=0.95, top=0.965, bottom=0.03, wspace=0.3, hspace=0.09)

    ax = plt.subplot(the_grid[0, 0], aspect=1)
    plt.pie(train_label_dist, labels=expr_classes, autopct='%1.1f%%', startangle=90, radius=1, pctdistance=0.85)
    plt.title('Training Set')
    plt.subplot(the_grid[0, 1], aspect=1)
    plt.pie(valid_label_dist, labels=expr_classes, autopct='%1.1f%%', startangle=90, radius=1, pctdistance=0.85)
    plt.title('Validation Set')
    plt.subplot(the_grid[0, 2], aspect=1)
    plt.pie(test_label_dist, labels=expr_classes, autopct='%1.1f%%', startangle=90, radius=1, pctdistance=0.85)
    plt.title('Test Set')
    plt.show()


def train_model(train_video_paths, train_label_paths,
                valid_video_paths, valid_label_paths,
                test_video_paths, test_label_paths,
                model_output_file, train_batch_size):

    # compute mean image
    img_size = (48, 48)    # size of image (h, w)
    composed_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])
    img_mean, img_std = calc_img_dataset_mean_std(train_video_paths, composed_tf)

    dataset_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
                                     ImgMeanStdNormalization(img_mean, img_std)])

    # train and test dataloader
    train_dataset = SFERDataset(train_video_paths, train_label_paths, transform=dataset_tf)
    train_dataloaer = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=SFERListCollate())

    valid_dataset = SFERDataset(valid_video_paths, valid_label_paths, transform=dataset_tf)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=SFERListCollate())

    test_dataset = SFERDataset(test_video_paths, test_label_paths, transform=dataset_tf)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=SFERListCollate())

    # model = Simple_LSTM()
    model = SFER_LSTM()
    weight = torch.FloatTensor([0, 0.137, 0.059, 0.181, 0.077, 0.21, 0.089, 0.247])
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = []
    all_valid_loss = []
    all_valid_acc = []
    all_test_acc = []
    lowest_valid_loss = 10
    q_len = 8
    change_acc = 0.6
    recent_valid_acc = collections.deque([0] * q_len, maxlen=q_len)

    epochs = 80

    start_train_time = time.time()
    for epoch in range(epochs):

        for i_batch, sample_batched in enumerate(train_dataloaer):
            model.train()

            optimizer.zero_grad()
            output, sorted_idx = model(sample_batched)
            target = Variable(torch.LongTensor([sample_batched[1][idx] for idx in sorted_idx]))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss = loss.data[0]

            all_train_loss.append(running_loss)

            valid_loss = 0.0
            is_best = False
            i = 0

            for i, sample_batched in enumerate(valid_dataloader, 1):
                model.eval()
                output, sorted_idx = model(sample_batched)
                target = Variable(torch.LongTensor([sample_batched[1][idx] for idx in sorted_idx]))
                loss = criterion(output, target)
                valid_loss += loss.data[0]

            if valid_loss / i < lowest_valid_loss:
                is_best = True
                lowest_valid_loss = valid_loss / i

            save_checkpoint({'epoch': epoch + 1, 'iteration': i_batch+1, 'state_dict': model.state_dict(),
                             'valid_loss': lowest_valid_loss, 'train_loss': running_loss}, is_best, model_output_file)

            all_valid_loss.append(valid_loss / i)

            valid_acc = eval_results(model, valid_dataloader)
            recent_valid_acc.append(valid_acc)
            all_valid_acc.append(valid_acc)

            test_acc = eval_results(model, test_dataloader)
            all_test_acc.append(test_acc)

            print('[%d, %5d] train loss: %.3f valid loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss, valid_loss / i))
            print('validation accuracy {}, test accuracy {}'.format(valid_acc, test_acc))
            print('lowest valid loss: ', lowest_valid_loss)
            print('\n')

            if sum(recent_valid_acc) / q_len > change_acc:
                print('change weights')
                for idx in range(1, weight.size()[0]):
                    if weight[idx] < 0.1:
                        weight[idx] = 0.1
                criterion = nn.CrossEntropyLoss(weight=weight)
                print(weight)
                lowest_valid_loss += 0.1
                print('lowest valid loss: ', lowest_valid_loss)
                change_acc = 1

    elapsed_train_time = time.time() - start_train_time
    print('Finish training. Spend {} minutes.'.format(elapsed_train_time / 60))

    # test results
    checkpoint = torch.load(model_output_file)
    model.load_state_dict(checkpoint['state_dict'])

    print("=>load checkpoint 'epoch: {}, iteration: {}, train_loss: {}, valid_loss: {}'".format(checkpoint['epoch'],
                                                                                               checkpoint['iteration'],
                                                                                               checkpoint['train_loss'],
                                                                                               checkpoint['valid_loss']))

    # check results on validation dataset
    valid_acc = eval_results(model, valid_dataloader, disp=True)
    print('Results on validation dataset {}'.format(valid_acc))

    # results on test dataset
    test_acc = eval_results(model, test_dataloader, disp=True)
    print('Results on test dataset {}'.format(test_acc))

    # plot losses
    iteration = list(range(len(all_train_loss)))
    plt.plot(iteration, all_train_loss, 'r', label='training loss')
    plt.plot(iteration, all_valid_loss, 'b', label='validation loss')
    plt.plot(iteration, all_valid_acc, 'y', label='validation accuracy')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    video_root_dir = r'/home/young/cv_project/CK_processed/cohn-kanade-images'

    label_root_dir = r'/home/young/cv_project/CK_processed/Emotion'

    train_file = './train_data2.txt'
    valid_file = './valid_data2.txt'
    test_file = './test_data2.txt'

    # video_dir_paths, label_dir_paths = get_ck_data(video_root_dir, label_root_dir)
    # split_datset_same_dist(video_dir_paths, label_dir_paths, 0.8)

    train_v_paths, train_l_paths, valid_v_paths, valid_l_paths, test_v_paths, test_l_paths = gen_vl_paths(train_file, valid_file, test_file)
    print('size of test data: ', len(test_v_paths))
    print('size of train data: ', len(train_v_paths))
    print('size of valid data: ', len(valid_v_paths))

    expr_classes = ('neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')

    trained_model_dir = r'/home/young/PycharmProjects/FER_DL/trained_models'
    # model_name = 'simple_lstm'
    model_name = 'sfer_lstm'
    model_idx = 2
    test_idx = 8
    model_file = os.path.join(trained_model_dir, model_name + str(model_idx) + '_' + str(test_idx) + '.pth.tar')

    train_batch_size = 64

    train_model(train_v_paths, train_l_paths,
                valid_v_paths, valid_l_paths,
                test_v_paths, test_l_paths,
                model_file, train_batch_size)
