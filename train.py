import argparse
import os

import torch.autograd
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

from data import data_helper
from data.data_gen_MNIST import adam
from data.data_helper import available_datasets
from models import model_factory
from models.caffenet import caffenet
from models.lenet import LeNet5
from models.resnet import resnet18
from utils.Logger import Logger
from utils.contrastive_loss import SupConLoss
from utils.util import *


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=10001, help="Number of epochs")
    parser.add_argument("--seen_index", "-s", type=int, default=0, help="Number of epochs")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet18")
    parser.add_argument("--aug_number", type=int, default=0, help="")
    parser.add_argument("--aug", type=str, default="", help="")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--imbalanced_class", type=bool, default=False,
                        help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--loops_min", type=int, default=200, help="")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--visualization", default=False, type=bool)
    parser.add_argument("--epochs_min", type=int, default=1,
                        help="")
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--ckpt", default="logs/model", type=str)
    #
    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr_sc", default=10, type=float)
    parser.add_argument("--task", default='PACS', type=str)

    return parser.parse_args()


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(SimpleDiscriminator, self).__init__()

        # Define the architecture
        self.net = nn.Sequential(
            # First linear layer
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Second linear layer
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class MixupGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, feat_size, num_classes):
        super(MixupGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)  # Assuming the output represents a probability distribution
        )

        # Linear transformations for Q, K, V
        self.W_Q = nn.Linear(feat_size, feat_size * 3, bias=False)
        self.W_K = nn.Linear(feat_size, feat_size, bias=False)
        self.W_V = nn.Linear(feat_size, feat_size, bias=False)
        self.sqrt_d = torch.sqrt(torch.tensor(feat_size, dtype=torch.float32))
        self.attn_drop = nn.Dropout(0.2)
        self.proj = nn.Linear(feat_size, feat_size)
        self.proj_drop = nn.Dropout(0.2)

        self.feature_size = feat_size
        self.num_classes = num_classes

    def entropy(self, p):
        p_clamped = p.clamp(min=1e-8, max=1 - 1e-8)
        return -(p_clamped * p_clamped.log()).sum(dim=-1)

    def correlation_matrix(self, Z):

        # Subtract the mean of each feature from the feature vector (mean centering)
        Z_centered = Z - Z.mean(dim=0)

        # Calculate the standard deviation of each feature
        std_dev = Z_centered.std(dim=0, unbiased=True)

        # Normalize each feature (mean-0 and std-1)
        Z_normalized = Z_centered / std_dev

        # Compute the correlation matrix by multiplying the normalized matrix transpose with itself
        correlation_mat = torch.matmul(Z_normalized.t(), Z_normalized) / (Z.size(0) - 1)

        # Make all values non-negative
        correlation_mat_non_neg = torch.abs(correlation_mat)

        # Normalize each row to make it a probability vector
        row_sums = correlation_mat_non_neg.sum(dim=1, keepdim=True)
        probability_matrix = correlation_mat_non_neg / row_sums

        return probability_matrix

    def forward(self, Z, y_lb=None, query=None, feature_wise=False):
        # Calculate batch size and feature size
        b, k, feat_size = Z.shape

        # Process Z through layers to get epsilon, reshaped to match Z's batch structure
        entropies = self.entropy(Z)

        epsilon = self.layers(entropies)

        a_expanded = epsilon.unsqueeze(-1)  # Shape: [b, k, 1]

        # Perform batched matrix multiplication and then squeeze the last dimension
        z_epsilon = (a_expanded * Z).sum(dim=1).view(b, 1, -1)
        # Z_centerlize = Z - Z.mean(dim=1, keepdim=True)
        # Z=Z_centerlize/Z.std(dim=1, keepdim=True)
        if query is not None:
            # query_centerlize = query - query.mean(dim=1, keepdim=True)
            # query_centerlize = query_centerlize / query.std(dim=1, keepdim=True)
            query = query.view(b, 1, -1)
            q = self.W_Q(query)
        else:
            q = self.W_Q(z_epsilon)

        if feature_wise:
            C = self.correlation_matrix(Z.view(b * k, feat_size))
            # Apply diagonalization of C and compute q_j and K_j for all j
            C_diag = torch.diag_embed(C)  # Expand C to a batch of diagonal matrices (d x d x d)
            q = self.W_Q(torch.einsum('bik,klk->bil', q, C_diag))  # (b, 1, d)
            K = self.W_Q(torch.einsum('bik,klk->bil', Z, C_diag))  # (b, k, d)
            V = self.W_Q(Z)  # (b, k, d)

            # Compute attention across all features in parallel
            A = F.softmax(torch.einsum('bid,bkd->bik', q, K) / self.sqrt_d, dim=-1)  # (b, 1, k)
            A = self.attn_drop(A)

            Z_aug = torch.matmul(A, V)  # (b, 1, d)
            Z_aug = self.proj(Z_aug)
            Z_aug = self.proj_drop(Z_aug)
            if y_lb is not None:
                Y_aug = torch.matmul(A, y_lb)  # (b, 1, c)
                return Z_aug, F.softmax(Y_aug, dim=-1)
            else:
                return Z_aug
        else:
            K = self.W_Q(Z)
            V = self.W_V(Z)  # Using Z directly as values

            # Compute scaled dot-product attention within each batch
            d_k = torch.tensor(self.feature_size, dtype=torch.float32).sqrt()
            attention_scores = torch.matmul(q, K.transpose(-2, -1)) / d_k
            attention_weights = F.softmax(attention_scores, dim=-1)
            z_aug = torch.matmul(attention_weights, V)

            # Assuming y_lb is provided and correctly batched as [b, k, num_classes]
            if y_lb is not None:
                # Apply epsilon as mixing coefficients to mix labels within each batch
                # This simplistic approach may need refinement for actual mixup behavior

                y_aug = torch.matmul(attention_weights, y_lb)
                return z_aug.view(b, -1), y_aug.view(b, -1)

        return z_aug.view(b, -1)


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.counterk = 0
        self.k = 5
        if args.task == 'digits':
            dim = 1024
        elif args.task == 'PACS':
            dim = 512

        self.mixup_generator = MixupGenerator(self.k, self.k * 2, self.k, dim, args.n_classes).cuda(self.args.gpu)

        self.discriminator = SimpleDiscriminator(dim).cuda(self.args.gpu)
        if args.task == 'digits':
            self.mixup_generator_optimizer = torch.optim.Adam(self.mixup_generator.parameters(), lr=0.0001,
                                                              weight_decay=1e-3)
        else:
            self.mixup_generator_optimizer = torch.optim.SGD(self.mixup_generator.parameters(), lr=0.001,
                                                             weight_decay=1e-3,
                                                             momentum=0.9,
                                                             nesterov=True, )
        if args.task == 'digits':
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001,
                                                            weight_decay=1e-3)
        else:
            self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=0.001, weight_decay=1e-3,
                                                           momentum=0.9,
                                                           nesterov=True, )
        self.discriminator.train()
        self.mixup_generator.train()

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        if args.task == 'PACS':
            self.extractor = resnet18(classes=args.n_classes, c_dim=512).to(device)
        elif args.task == 'digits':
            self.extractor = LeNet5(num_classes=args.n_classes).to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=False)
        if len(self.args.target) > 1:
            self.target_loader = data_helper.get_multiple_val_dataloader(args, patches=False)
        else:
            self.target_loader = data_helper.get_val_dataloader(args, patches=False)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        classifier_param = list(map(id, self.extractor.class_classifier.parameters()))
        backbone_param = filter(lambda p: id(p) not in classifier_param and p.requires_grad,
                                self.extractor.parameters())
        parameter_list = []
        parameter_list.append({'params': backbone_param, 'lr': 0.1 * self.args.learning_rate})

        parameter_list.append({'params': self.extractor.class_classifier.parameters(), 'lr': self.args.learning_rate})
        # Get optimizers and Schedulers, self.discriminator
        if args.task == 'PACS':
            self.optimizer = torch.optim.SGD(parameter_list, momentum=0.9,
                                             nesterov=True, weight_decay=0.00005)
        elif args.task == 'digits':
            self.optimizer = adam(
                parameters=self.extractor.parameters(),
                lr=1e-4,
                weight_decay=0,
            )

        if args.task == 'PACS':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs)
        elif args.task == 'digits':
            self.scheduler = lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=10001, gamma=0.1
            )
        self.scheduler2 = optim.lr_scheduler.CosineAnnealingLR(self.mixup_generator_optimizer, self.args.epochs)
        self.scheduler3 = optim.lr_scheduler.CosineAnnealingLR(self.discriminator_optimizer, self.args.epochs)

        self.n_classes = args.n_classes

        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        self.con = SupConLoss()
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    @torch.no_grad()
    def random_sample(self, feats, labels=None, labeled=True):
        N = feats.size(0)  # Total number of samples
        b = N // self.k
        # Randomly select b*k indices
        random_indices = torch.randperm(N)[:b * self.k]

        # Select the corresponding features and labels
        selected_feats = feats[random_indices]
        # Reshape to have them in [b, k, feat_size] for feats and [b, k] for labels
        selected_feats = selected_feats.view(b, self.k, -1)
        if labels is None:
            return selected_feats
        selected_labels = labels[random_indices]
        selected_labels = selected_labels.view(b, self.k, -1)
        return selected_feats, selected_labels

    def calculate_dis_loss(self, feat, query=None):
        criterion = nn.BCELoss()

        # Forward pass for real data

        real_outputs = self.discriminator(feat.detach())
        real_labels = torch.ones_like(real_outputs)  # Real data label is 1
        real_loss = criterion(real_outputs, real_labels)

        # Forward pass for fake data
        selected_feats = self.random_sample(feat)
        z_aug = self.mixup_generator(selected_feats.detach(), query=query)
        fake_outputs = self.discriminator(z_aug)
        fake_labels = torch.zeros_like(fake_outputs)  # Fake data label is 0

        fake_loss = criterion(fake_outputs, fake_labels)

        # Total loss
        return real_loss + fake_loss

    def _l2_normalize(self, d):
        # TODO: put this to cuda with torch
        d = d.numpy()
        if len(d.shape) == 4:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        elif len(d.shape) == 3:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def kl_div_with_logit(self, q_logit, p_logit):

        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)

        return qlogq - qlogp

    def calculate_gen_loss(self, feat, query=None):
        criterion = nn.BCELoss()
        selected_feats = self.random_sample(feat)
        z_aug = self.mixup_generator(selected_feats.detach(), query=query)
        with torch.no_grad():
            fake_outputs = self.discriminator(z_aug)
        fake_labels = torch.ones_like(fake_outputs)  # Fake data label is 1
        fake_loss = criterion(fake_outputs, fake_labels)

        # Total loss
        return fake_loss

    def _do_epoch(self, epoch=None):
        self.extractor.train()
        self.bn_eval(self.extractor)
        self.mixup_generator.train()
        self.discriminator.train()
        for it, ((data, _, y_lb), _, idx) in enumerate(self.source_loader):
            data, y_lb = data.to(self.device), y_lb.to(self.device)
            self.mixup_generator_optimizer.zero_grad()
            with  torch.no_grad():
                logits, tuples = self.extractor(x=data)
            feats_x_lb = tuples['Embedding']
            N = feats_x_lb.size(0)  # Total number of samples
            b = N // self.k
            query = self.get_queries(self.extractor, data[:b], y_lb[:b])
            with  torch.no_grad():
                logits_q, tuples_q = self.extractor(x=query)
            self.discriminator_optimizer.zero_grad()
            query_embadding = tuples_q['Embedding']
            dis_loss = self.calculate_dis_loss(feats_x_lb, query_embadding)
            dis_loss.backward()
            self.discriminator_optimizer.step()
            adv_loss = self.calculate_gen_loss(feats_x_lb, query_embadding)

            # Inner optimization step for phi (augmentation model)

            one_hot_lb = F.one_hot(y_lb, num_classes=self.n_classes).to(torch.float32).to(self.device)
            selected_feats, selected_y_lb = self.random_sample(feats_x_lb, one_hot_lb)
            z_aug, y_aug = self.mixup_generator(selected_feats, selected_y_lb, query=query_embadding)
            with torch.no_grad():
                logits = self.extractor(z_aug, classify=True)
            loss_aug_main = F.cross_entropy(logits, y_aug, reduction='mean')
            loss = adv_loss + loss_aug_main
            loss.backward()
            self.mixup_generator_optimizer.step()

        for it, ((data, _, y_lb), _, idx) in enumerate(self.source_loader):
            data, y_lb = data.to(self.device), y_lb.to(self.device)
            # ----------------------
            # Stage 2
            self.optimizer.zero_grad()

            logits, tuples = self.extractor(x=data)
            N = tuples['Embedding'].size(0)  # Total number of samples
            b = N // self.k
            query = self.get_queries(self.extractor, data[:b], y_lb[:b])
            logits_q, tuples_q = self.extractor(x=query)
            query_embadding = tuples_q['Embedding']
            sup_loss = F.cross_entropy(logits, y_lb, reduction='mean')

            feats_x_lb = tuples['Embedding']

            one_hot_lb = F.one_hot(y_lb, num_classes=self.n_classes).to(torch.float32).to(self.device)
            #
            with torch.no_grad():
                selected_feats, selected_y_lb = self.random_sample(feats_x_lb, one_hot_lb)
                z_aug, y_aug = self.mixup_generator(selected_feats, selected_y_lb, query_embadding)
            loss_aug_main = F.cross_entropy(self.extractor(z_aug, classify=True), y_aug, reduction='mean')
            unsup_warmup = np.clip(epoch / self.args.epochs, a_min=0.0, a_max=1.0)
            # mixup_data, mixup_label, lam = self.mixup_one_target(data, y_lb, alpha=0.75, is_bias=True)
            # logits, touple = self.extractor(mixup_data)
            # mixup_loss = F.cross_entropy(logits, mixup_label.type(torch.LongTensor).to(self.device), reduction='mean')
            loss = (sup_loss + 0.5 * loss_aug_main)

            loss.backward()
            self.optimizer.step()

        del loss, logits
        if epoch % 50 == 0:
            self.extractor.eval()
            with torch.no_grad():

                if len(self.args.target) > 1:
                    avg_acc = 0
                    for i, loader in enumerate(self.target_loader):
                        total = len(loader.dataset)

                        class_correct = self.do_test(loader)

                        class_acc = float(class_correct) / total
                        self.logger.log_test('test', {"class": class_acc})

                        avg_acc += class_acc
                    avg_acc = avg_acc / len(self.args.target)
                    print(avg_acc)
                    self.results["test"][self.current_epoch] = avg_acc
                else:
                    for phase, loader in self.test_loaders.items():
                        if self.args.task == 'HOME' and phase == 'val':
                            continue
                        total = len(loader.dataset)

                        class_correct = self.do_test(loader)

                        class_acc = float(class_correct) / total
                        self.logger.log_test(phase, {"class": class_acc})
                        self.results[phase][self.current_epoch] = class_acc

    @torch.no_grad()
    def mixup_one_target(self, x, y, alpha=1.0, is_bias=False):
        """Returns mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, nouse, class_l), _, _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            z = self.extractor(data, train=False)[0]

            _, cls_pred = z.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct

    def bn_eval(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        for self.current_epoch in tqdm(range(self.args.epochs)):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            self.scheduler2.step()
            self.scheduler3.step()
            if self.results["test"][self.current_epoch] > current_high:
                print('Saving Best model ...')
                torch.save(self.extractor.state_dict(), os.path.join('logs/model/', 'best_' + self.args.target[0]))
                current_high = self.results["test"][self.current_epoch]
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best test epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger

    def get_queries(self, model, x, y, lambda_val=1e-6, eta=1e-6, num_iters=1):
        x_hat = x.requires_grad_()
        for i in range(num_iters):
            logits, _ = model(x_hat)
            delta_kl = F.cross_entropy(logits, y, reduction='mean')
            delta_kl.backward()
            with torch.no_grad():
                noise = torch.randn_like(x) * lambda_val
                x_hat = x + eta * x.grad + noise  # Update rule
            model.zero_grad()
        return x_hat

    def _l2_normalize(self, d):
        # TODO: put this to cuda with torch
        d = d.numpy()
        if len(d.shape) == 4:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        elif len(d.shape) == 3:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def do_eval(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        self.logger.new_epoch(self.scheduler.get_lr())
        self.extractor.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][0] = class_acc

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best test epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger


def main():
    args = get_args()

    if args.task == 'PACS':
        args.n_classes = 7
        # args.source = ['art_painting', 'cartoon', 'sketch']
        # args.target = ['photo']
        # args.source = ['art_painting', 'photo', 'cartoon']
        # args.target = ['sketch']
        # args.source = ['art_painting', 'photo', 'sketch']
        # args.target = ['cartoon']
        # args.source = ['photo', 'cartoon', 'sketch']
        # args.target = ['art_painting']
        # --------------------- Single DG
        args.source = ['photo']
        args.target = ['cartoon', 'sketch', 'art_painting']

    elif args.task == 'VLCS':
        args.n_classes = 5
        # args.source = ['CALTECH', 'LABELME', 'SUN']
        # args.target = ['PASCAL']
        args.source = ['LABELME', 'SUN', 'PASCAL']
        args.target = ['CALTECH']
        # args.source = ['CALTECH', 'PASCAL', 'LABELME' ]
        # args.target = ['SUN']
        # args.source = ['CALTECH', 'PASCAL', 'SUN']
        # args.target = ['LABELME']

    elif args.task == 'HOME':
        args.n_classes = 65
        # args.source = ['real', 'clip', 'product']
        # args.target = ['art']
        # args.source = ['art', 'real', 'product']
        # args.target = ['clip']
        # args.source = ['art', 'clip', 'real']
        # args.target = ['product']
        args.source = ['art', 'clip', 'product']
        args.target = ['real']
    elif args.task == 'cifar10-c':
        args.n_classes = 10
        args.source = ['cifar10']
        args.target = ['cifar10', ]

    elif args.task == 'digits':
        args.n_classes = 10
        args.source = ['mnist']
        args.target = ['usps', 'svhn', 'mnist_m', 'syndigit']

    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    fix_all_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    if not args.eval:
        trainer.do_training()
    else:
        # trainer.extractor.load_state_dict(torch.load(''))
        trainer.do_eval()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
