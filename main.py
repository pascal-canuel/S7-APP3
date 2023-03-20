from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *


def visualize_attention(target, prediction, attention, handwritten):
    if len(target) != len(target) != attention.shape[1]:
        raise ValueError("target and prediction must have the same length")

    word_len = len(target)

    fig, axes = plt.subplots(word_len, 1)

    for idx in range(word_len):
        axes[idx].cla()

        attn = attention[:, idx]
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        attn = 1 - attn

        axes[idx].scatter(handwritten[0], handwritten[1], c=attn, cmap='gray')
        axes[idx].plot(handwritten[0], handwritten[1], c='lightgray')

        axes[idx].set_ylabel(prediction[idx])
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.tight_layout()
    #plt.title(f'target: {" ".join(target)} \r\n prediction {" ".join(prediction)}')

    # make room for title
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f'target: {" ".join(target)} \n prediction {" ".join(prediction)}', fontsize=14)

    plt.show()


if __name__ == '__main__':
    training = True
    test = True
    learning_curves = True
    gen_test_images = True
    seed = 1
    n_workers = 0
    n_epochs = 400
    lr = 0.02
    batch_size = 64
    train_val_split = 0.8
    hidden_dim = 12
    n_layers = 2

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HandwrittenWords('data_train_val.p')

    n_train_samples = int(len(dataset) * train_val_split)
    n_val_samples = len(dataset) - n_train_samples

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)

    model = trajectory2seq(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        device=device,
        maxlen=dataset.max_len
    )

    print(f'Model : \n {model} \n')
    print(f'Weights number: {sum([i.numel() for i in model.parameters()])}')

    if training:
        train_dist = []
        train_loss = []

        val_dist = []
        val_loss = []

        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if learning_curves:
            fig, (ax1, ax2) = plt.subplots(1, 2)

        for epoch in range(1, n_epochs + 1):
            running_loss_train = 0
            dist_train = 0

            for batch_idx, data in enumerate(train_loader):
                handwritten_seq, target_seq = data
                handwritten_seq = handwritten_seq.to(device).float()
                target_seq = target_seq.to(device)

                optimizer.zero_grad()
                output, hidden, attn = model(handwritten_seq)
                loss = criterion(output.view((-1, model.dict_size['word'])), target_seq.view(-1))

                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                batch_len = len(output_list)
                for i in range(batch_len):
                    a = target_seq_list[i]
                    b = output_list[i]
                    a_len = a.index(1)
                    b_len = b.index(1) if 1 in b else len(b)
                    dist_train += edit_distance(a[:a_len], b[:b_len]) / batch_size

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch,
                n_epochs,
                (batch_idx + 1) * batch_size,
                len(train_loader.dataset),
                100. * (batch_idx + 1) * batch_size / len(train_loader.dataset),
                running_loss_train / (batch_idx + 1),
                dist_train / len(train_loader)), end='\r\n')

            train_loss.append(running_loss_train / len(train_loader))
            train_dist.append(dist_train / len(train_loader))

            running_loss_val = 0
            dist_val = 0

            for batch_idx, data in enumerate(val_loader):
                handwritten_seq, target_seq = data
                handwritten_seq = handwritten_seq.to(device).float()
                target_seq = target_seq.to(device)

                output, hidden, attn = model(handwritten_seq)
                loss = criterion(output.view((-1, model.dict_size['word'])), target_seq.view(-1))

                running_loss_val += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                batch_len = len(output_list)
                for i in range(batch_len):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    dist_val += edit_distance(a[:Ma], b[:Mb]) / batch_size

            print('Validation - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                 epoch,
                 n_epochs,
                 (batch_idx + 1) * batch_size,
                 len(val_loader.dataset),
                 100. * (batch_idx + 1) * batch_size / len(val_loader.dataset),
                 running_loss_val / (batch_idx + 1),
                 dist_val / len(val_loader)), end='\r\n')

            val_loss.append(running_loss_val / len(val_loader))
            val_dist.append(dist_val / len(val_loader))

            torch.save(model, 'model.pt')

            if learning_curves:
                ax1.cla()
                ax1.plot(train_loss, label='Training')
                ax1.plot(val_loss, label='Validation')
                ax1.legend()
                ax1.grid()
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss')
                ax1.set_title('Loss')

                ax2.cla()
                ax2.plot(train_dist, label='Training')
                ax2.plot(val_dist, label='Validation')
                ax2.legend()
                ax2.grid()
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Distance')
                ax2.set_title('Distance')

                plt.draw()
                plt.pause(0.01)

        if learning_curves:
            plt.show()

    if test:
        model = torch.load('model_lstm_bi_attention.pt')
        dataset_test = HandwrittenWords('data_train_val.p')

        targets = []
        outputs = []

        dt = dataset_test
        for i in range(len(dt)):
        # for i in range(100):
            handwritten_seq, target_seq = dt[i]
            handwritten_seq_nan = handwritten_seq
            handwritten_seq = handwritten_seq.to(device).float()
            target_seq = target_seq.to(device)

            handwritten_seq = torch.unsqueeze(handwritten_seq, dim=0)
            output, hidden, attn = model(handwritten_seq)

            output_list = torch.argmax(output, dim=-1).detach().cpu().squeeze().tolist()
            target_seq_list = target_seq.detach().cpu().tolist()

            output_seq_symb = [dt.int2symb['word'][c] for c in output_list]
            target_seq_symb = [dt.int2symb['word'][c] for c in target_seq_list]

            targets.append(target_seq_symb)
            outputs.append(output_seq_symb)

            visualize_attention(target_seq_symb, output_seq_symb, torch.squeeze(attn).detach().cpu().numpy(), torch.squeeze(handwritten_seq_nan).detach().cpu().numpy())
            print(f'Output: {" ".join(output_seq_symb)}; Target: {" ".join(target_seq_symb)}\r\n')

        labels = list(dt.symb2int['word'].keys())
        confusion_matrix(targets, outputs, labels)

        # TODO: display attention
        # TODO: print test results

        pass
