from seq_fer_datasets import *
from seq_fer import SFER_LSTM, Simple_LSTM
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch


def save_checkpoint(state_dict, is_best, filename):
    if is_best:
        print('=> Saving a new best')
        torch.save(state_dict, filename)
    else:
        print('=> Validation accuracy did not improve')


def simple_lstm_test(train_video_paths, train_label_paths, test_video_paths, test_label_paths, model_output_file, train_batch_size):

    # compute mean image
    img_size = (48, 48)    # size of image (h, w)
    composed_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])
    img_mean, img_std = calc_img_dataset_mean_std(train_video_paths, composed_tf)

    dataset_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
                                     ImgMeanStdNormalization(img_mean, img_std)])

    # train and test dataloader
    train_dataset = SFERDataset(train_video_paths, train_label_paths, transform=dataset_tf)
    train_dataloaer = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=SFERListCollate())
    # train_dataloaer = DataLoader(train_dataset, shuffle=True)

    test_dataset = SFERDataset(test_video_paths, test_label_paths, transform=dataset_tf)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=SFERListCollate())

    model = Simple_LSTM()
    model = SFER_LSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = []
    all_test_loss = []
    lowest_test_loss = 10

    for epoch in range(80):

        for i_batch, sample_batched in enumerate(train_dataloaer):

            # model(sample_batched)
            #
            # if i_batch == 0:
            #     break

            optimizer.zero_grad()
            output, sorted_idx = model(sample_batched)
            target = Variable(torch.LongTensor([sample_batched[1][idx] for idx in sorted_idx]))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss = loss.data[0]


            all_train_loss.append(running_loss)

            test_loss = 0.0
            is_best = False
            i = 0
            for i, sample_batched in enumerate(test_dataloader, 1):
                output, sorted_idx = model(sample_batched)
                target = Variable(torch.LongTensor([sample_batched[1][idx] for idx in sorted_idx]))
                loss = criterion(output, target)
                test_loss += loss.data[0]

            if test_loss < lowest_test_loss:
                is_best = True
                lowest_test_loss = test_loss

            save_checkpoint({'epoch': epoch + 1, 'iteration': i_batch+1, 'state_dict': model.state_dict(),
                             'test_loss': lowest_test_loss / i, 'train_loss': running_loss}, is_best, model_output_file)

            all_test_loss.append(test_loss)
            # print('test loss: %.3f' %
            #       (test_loss / i))

            print('[%d, %5d] train loss: %.3f test loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss, test_loss / i))

    print('Finish training')

    # test results
    expr_classes = ('neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')

    checkpoint = torch.load(model_output_file)
    model.load_state_dict(checkpoint['state_dict'])

    print("=>load checkpoint 'epoch: {}, iteration: {}, train_loss: {}, test_loss: {}'".format(checkpoint['epoch'],
                                                                                               checkpoint['iteration'],
                                                                                               checkpoint['train_loss'],
                                                                                               checkpoint['test_loss']))
    for epoch in range(1):
        for sample_batched in test_dataloader:
            output, sorted_idx = model(sample_batched)
            max_values, predicted = torch.max(output.data, 1)
            print('{:16}{}'.format('ground truth: ', ['{:8}'.format(expr_classes[sample_batched[1][idx]]) for idx in sorted_idx]))
            print('{:16}{}'.format('predict: ', ['{:8}'.format(expr_classes[pred]) for pred in predicted]))

    iteration = list(range(len(all_train_loss)))

    plt.plot(iteration, all_train_loss, 'r', label='train loss')
    plt.plot(iteration, all_test_loss, 'b', label='test_loss')
    plt.legend(loc='upper right')
    plt.xlabel('Check Iteration')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    video_root_dir = r'/home/young/cv_project/CK_processed/cohn-kanade-images'

    label_root_dir = r'/home/young/cv_project/CK_processed/Emotion'

    video_dir_paths, label_dir_paths = get_ck_data(video_root_dir, label_root_dir)

    entire_dataset_size = len(video_dir_paths)

    print('size of the dataset: ', entire_dataset_size)

    # split the entire dataset into train and test
    indices = list(range(entire_dataset_size))
    # random.shuffle(indices)

    # split_idx = int(entire_dataset_size * 0.9)

    train_v_paths = [video_dir_paths[idx] for idx in indices if idx % 9 != 0]
    train_l_paths = [label_dir_paths[idx] for idx in indices if idx % 9 != 0]

    test_v_paths = [video_dir_paths[idx] for idx in indices if idx % 9 == 0]
    test_l_paths = [label_dir_paths[idx] for idx in indices if idx % 9 == 0]

    trained_model_dir = r'/home/young/PycharmProjects/FER_DL/trained_models'
    # model_name = 'simple_lstm'
    model_name = 'sfer_lstm'
    model_idx = 1
    test_idx = 3
    model_file = os.path.join(trained_model_dir, model_name + str(model_idx) + '_' + str(test_idx) + '.pth.tar')

    train_batch_size = 64
    simple_lstm_test(train_v_paths, train_l_paths, test_v_paths, test_l_paths, model_file, train_batch_size)