from seq_fer_datasets import *
from seq_fer import SFER_LSTM
from torch.autograd import Variable


if __name__ == '__main__':
    video_root_dir = r'/home/young/cv_project/cohn-kanade-images'

    label_root_dir = r'/home/young/cv_project/Emotion'

    video_dir_paths, label_dir_paths = get_ck_data(video_root_dir, label_root_dir)

    img_size = (320, 240)
    composed_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])

    # img_mean, img_std = calc_img_dataset_mean_std(video_dir_paths, composed_tf)

    # dataset_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
    #                                  ImgMeanStdNormalization(img_mean, img_std)])

    sfer_dataset = SFERDataset(video_dir_paths, label_dir_paths, transform=composed_tf)

    sfer_dataloaer = DataLoader(sfer_dataset, batch_size=8, shuffle=True, collate_fn=SFERPadCollate(dim=0))

    model = SFER_LSTM()

    for i_batch, sample_batched in enumerate(sfer_dataloaer):
        # print(type(sample_batched))
        # for v in sample_batched[0]:
        #     print(v.size())
        #
        # print(sample_batched[0].size())

        videos, labels = sample_batched
        # videos, labels = Variable(videos), Variable(labels)

        model(sample_batched)

        if i_batch == 0:
            break