from seq_fer_datasets import *
from seq_fer import SFER_LSTM, Simple_LSTM
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch


if __name__ == '__main__':
    video_root_dir = r'/home/young/cv_project/CK_processed/cohn-kanade-images'

    label_root_dir = r'/home/young/cv_project/CK_processed/Emotion'

    video_dir_paths, label_dir_paths = get_ck_data(video_root_dir, label_root_dir)

    # size of image (h, w)
    img_size = (48, 48)
    composed_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])

    img_mean, img_std = calc_img_dataset_mean_std(video_dir_paths, composed_tf)

    dataset_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
                                     ImgMeanStdNormalization(img_mean, img_std)])

    # sfer_dataset = SFERDataset(video_dir_paths, label_dir_paths, transform=composed_tf)
    sfer_dataset = SFERDataset(video_dir_paths, label_dir_paths, transform=dataset_tf)

    # sfer_dataloaer = DataLoader(sfer_dataset, batch_size=8, shuffle=True, collate_fn=SFERPadCollate(dim=0))
    # sfer_dataloaer = DataLoader(sfer_dataset, batch_size=8, shuffle=True, collate_fn=SFERListCollate())
    sfer_dataloaer = DataLoader(sfer_dataset, shuffle=False)

    # model = SFER_LSTM()
    model = Simple_LSTM()
    # model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        running_loss = 0.0

        for i_batch, sample_batched in enumerate(sfer_dataloaer):

            optimizer.zero_grad()
            output = model(Variable(sample_batched[SAMPLE_INPUT]))
            # print(torch.LongTensor(sample_batched[SAMPLE_TARGET]))
            loss = criterion(output, Variable(torch.LongTensor(sample_batched[SAMPLE_TARGET])))
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i_batch % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0

    print('Finish training')



    # for epoch in range(10):
    #     running_loss = 0.0
    #
    #     for i_batch, sample_batched in enumerate(sfer_dataloaer):
    #         # print(type(sample_batched))
    #         # for v in sample_batched[0]:
    #         #     print(len(v))
    #         #     print(v[0].size())
    #         # print(type(sample_batched))
    #         # for v in sample_batched[0]:
    #         #     print(v.size())
    #         #
    #         # print(sample_batched[0].size())
    #
    #         # videos, labels = sample_batched
    #         # videos, labels = Variable(videos), Variable(labels)
    #
    #         optimizer.zero_grad()
    #         out = model(sample_batched)
    #         # print(torch.LongTensor(sample_batched[1]))
    #         loss = criterion(out, Variable(torch.LongTensor(sample_batched[1])))
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.data[0]
    #
    #         if i_batch % 10 == 0:
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i_batch + 1, running_loss / 10))
    #             running_loss = 0.0
    #
    # print('Finish training')