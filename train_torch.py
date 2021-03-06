from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from args import args
from data import EnumPairedDataset
from data import train_val_dataset
from torchsummary import summary
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import math
from logs import add_line_to_csv
import torch
import piq
from models import models
from loss import loss as loss_fun


model_names = ",".join(list(models.keys()))

def main(args):

    input_shape = (3, 380, 380)
    if not os.path.exists(args.checkpoints_output):
        os.makedirs(args.checkpoints_output)

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)

    images_output = os.path.join(args.logs, 'images')
    if not os.path.exists(images_output):
        os.makedirs(images_output)

    if not args.model in models:
        print(f"Model name {args.model} must be one of: {model_names}")
        return 1

    print(f"Seting up training for model: {args.model}")
    print(f"Train X Root: {args.train_x_root}")
    print(f"Train Y Root: {args.train_y_root}")

    if args.test_x is not None and args.test_y is not None:
        print(f"Test X Root: {args.test_x}")
        print(f"Test Y Root: {args.test_y}")

    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                 0.5))
    if args.test_x is None or args.test_y is None:
        dataset = EnumPairedDataset(args.train_x_root, args.train_y_root, transform=normalize_transform)
        train_d, test_d = train_val_dataset(dataset)
    else:
        train_d = EnumPairedDataset(args.train_x_root, args.train_y_root, transform=normalize_transform)
        test_d = EnumPairedDataset(args.test_x_root, args.test_y_root, transform=normalize_transform)

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_dl = DataLoader(train_d, batch_size=train_batch_size,
                                                shuffle=True, num_workers=0)
    test_dl = DataLoader(test_d, batch_size=test_batch_size,
                                                shuffle=True, num_workers=0)

    if args.show_dataset:
        x_batch, y_batch, names = next(iter(train_dl))
        plt.subplot(2, 1, 1)
        plt.imshow(torchvision.utils.make_grid(x_batch).permute(1, 2, 0))
        plt.subplot(2, 1, 2)
        plt.imshow(torchvision.utils.make_grid(y_batch).permute(1, 2, 0))
        plt.show()

    model = models[args.model]

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    device = f"cuda:{model.device_ids[0]}"
    #device = 'cpu'
    model.to(device)
    summary(model, input_shape)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #criterion = loss_fun()

    if args.load is not None:
        try:
            pretrained_dict = torch.load(args.load)
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Weights loaded from {args.load}")
        except Exception as e:
            print(f"Couldn't load weights from {args.load}")
            print(e)

    best_training_loss = math.inf
    test_loss_for_best_training_loss = math.inf

    cols = ['epoch', 'training_loss', 'test_loss', 'train_psnr', 'test_psnr',
            'train_ssim', 'test_ssim']
    logs_addr = os.path.join(args.logs, 'logs.csv')
    add_line_to_csv(logs_addr, cols)

    print(f"Logs to: {args.logs}")
    epochs = args.epochs
    if epochs <= args.from_epoch:
        epochs += args.from_epoch

    for epoch in range(args.from_epoch, epochs + 1):

        print(f"Epoch {epoch}/{epochs}")

        training_loss = 0.0
        test_loss = 0.0

        train_psnr = 0.0
        test_psnr = 0.0
        train_ssim = 0.0
        test_ssim = 0.0

        if args.load is not None:
            try:
                fn, ext = os.path.splitext(os.path.basename(args.load))
                loss_vals = fn.split('_')
                best_training_loss = float(loss_vals[2])
                test_loss_for_best_training_loss = float(loss_vals[3])
            except Exception as e:
                print(f"Couldn't load best training loss from {args.load}")
                print(e)

        print("Training:")
        for i, data in tqdm(enumerate(train_dl), total=int(len(train_d)
                            / train_batch_size)):
            w, m, file_name = data
            x = w.to(device)
            y = m.to(device)
            del w
            del m

            optimizer.zero_grad()
            y_hat = model(x)

            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()

            training_loss += float(loss.item())
            del x

            '''
            train_psnr = piq.psnr(y_hat[0], y[0],data_range=1.,
                                   reduction='none')
            train_ssim = piq.ssim(y_hat[0], y[0], data_range=1.,
                                   reduction='none')
            '''

            del y
            del y_hat

        training_loss /= (i+1)
        #train_psnr /= (i+1)
        #train_ssim /= (i+1)

        with torch.no_grad():
            print("Testing:")
            for i, data in tqdm(enumerate(test_dl), total=int(len(test_d)
                                                              / test_batch_size)):
                w, m, file_name = data
                x = w.to(device)
                y = m.to(device)
                y_hat = model(x)
                loss = loss_fun(y_hat, y)
                test_loss += float(loss.item())
                del x

                try:
                    test_psnr += piq.psnr(y_hat, y)
                    test_ssim += piq.ssim(y_hat, y)
                except:
                    pass

                if args.show_output_images and i < 5:
                    imgs_dir = os.path.join(images_output, f"epoch_{epoch}")
                    if not os.path.exists(imgs_dir):
                        os.makedirs(imgs_dir)
                    for j, y_hat_i in enumerate(y_hat):
                        fn = os.path.splitext(os.path.basename(file_name[j]))[0]
                        y_gt = y[j]
                        img_i_addr = os.path.join(imgs_dir, f'{epoch}_{fn}_{i}_{j}.{dataset.images_extension}')
                        img_i_gt_addr = os.path.join(imgs_dir, f'{epoch}_{fn}_{i}_{j}_gt.{dataset.images_extension}')
                        torchvision.utils.save_image(y_hat_i, img_i_addr)
                        torchvision.utils.save_image(y_gt, img_i_gt_addr)
                        del y_gt
                del y
                del y_hat

        test_loss /= (i+1)
        test_psnr /= (i + 1)
        test_ssim /= (i + 1)

        print(f"Completed Epoch: {epoch}/{args.epochs}")
        print(f"\tTrain loss: {training_loss}")
        print(f"\tTest loss: {test_loss}")
        print(f"\tTrain PSNR: {train_psnr}")
        print(f"\tTest PSNR: {test_psnr}")
        print(f"\tTrain SSIM: {train_ssim}")
        print(f"\tTest SSIM: {test_ssim}")
        print(f"\tBest training loss so far: {best_training_loss}")
        print(f"\tTest loss for: {test_loss_for_best_training_loss}")

        add_line_to_csv(logs_addr, [str(epoch), str(training_loss),
                                    str(test_loss), str(train_psnr),
                                    str(test_psnr), str(train_ssim),
                                    str(test_ssim)])

        if best_training_loss > training_loss:
            best_training_loss = training_loss
            test_loss_for_best_training_loss = test_loss
            save_file_name = f"{args.model}_epoch_{epoch}_{best_training_loss:.3f}_{test_loss_for_best_training_loss:.3f}.pth"
            checkpoint_path = os.path.join(args.checkpoints_output, save_file_name)
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main(args)
