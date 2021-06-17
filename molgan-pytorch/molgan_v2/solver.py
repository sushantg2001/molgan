# %%
import numpy as np
import os
import time
import datetime
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import *
from models import Generator, Discriminator
from sparse_molecular_dataset import SparseMolecularDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = 'unique,validity'
        # self.metric = 'validity,unique,novelty,odor'
        # Training configurations.
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.graph_dir = config.graph_dir
        self.m1_result_dir = config.m1_result_dir
        self.generate_dir = config.generate_dir
        self.m1_output = config.m1_output
        self.m1_input = config.m1_input
        self.metric_scores = config.metric_scores
        self.m1_dir = config.m1_dir
        self.odor_model = config.odor_model
        self.loss_dir = config.loss_dir
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)

        self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.V.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        return out

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            elif m == 'odor':
                rr *= MolecularMetrics.odor(mols,self.m1_output,self.m1_input)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def smiles_to_mols(self, mols):
        temp = []
        for i in mols:
            if i is not None:
                test = Chem.MolToSmiles(i)
                if test is not None:
                    temp.append(test)
        return temp

    def print_smiles(self,mols):
        temp = self.smiles_to_mols(mols)
        print(temp)
        return temp

    def generate_smiles(self, mols, generate_dir):
        path = generate_dir
        if not os.path.exists(path):
            os.makedirs(path)

        temp = self.print_smiles(mols)

        df = pd.DataFrame(temp, columns=['smiles'])
        name = datetime.datetime.now()
        df.to_csv(
            f'{path}/{name}.csv', index=False)
        return name


    def train(self):

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            print(i+1)
            if (i+1) % self.log_step == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
                print('[Valid]', '')
            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(
                    self.batch_size)
                z = self.sample_z(self.batch_size)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # self.generate_smiles(mols,self.m1_input)
            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            # print(a_tensor,'a_tensor')
            # print(x_tensor, 'x_tensor')
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real = - torch.mean(logits_real)

            # Compute loss with fake images.
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            d_loss_fake = torch.mean(logits_fake)

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)


            # Backward and optimize.
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            # print("Discriminator","d_loss_fake",d_loss_fake ,"d_loss_real", d_loss_real ,"Gradpen loss ", self.lambda_gp * d_loss_gp)
            self.reset_grad()
            f = open(f'{self.loss_dir}/discriminator_loss.txt', 'a+')
            f.write(f'{d_loss}\n')
            d_loss.backward()
            self.d_optimizer.step()

            print("Discriminator Loss",d_loss)

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)
                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
                g_loss_fake = - torch.mean(logits_fake)

                # Real Reward
                rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
                # Fake Reward
                (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
                edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
                mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                        for e_, n_ in zip(edges_hard, nodes_hard)]
                self.print_smiles(mols)
                rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

                # Value loss
                value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
                g_loss_value = torch.mean((value_logit_real - rewardR) ** 2 + (
                                           value_logit_fake - rewardF) ** 2)
                #rl_loss= -value_logit_fake
                #f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_value
                # print("generator loss","g_loss_fake",g_loss_fake,"g_loss_real",g_loss_value)
                self.reset_grad()
                f = open(f'{self.loss_dir}/generator_loss.txt', 'a+')
                f.write(f'{d_loss}\n')
                g_loss.backward()
                self.g_optimizer.step()

                print("generator loss",g_loss)

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_value'] = g_loss_value.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i+1, self.num_iters)

                # Log update
                # 'mols' is output of Fake Reward
                m0, m1 = all_scores(mols, self.m1_output,
                                    self.m1_input, self.data, norm=True)
                m0 = {k: np.array(v)[np.nonzero(v)].mean()
                      for k, v in m0.items()}
                m0.update(m1)
                f = open(
                    f'{self.metric_scores}/{datetime.datetime.now()}.txt', 'w+')
                f.write(f"{i+1} iteration\n")
                f.write(str(m0))
                print(m0)
                loss.update(m0)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
            # if (i+1) % self.lr_update_step == 0:
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))



    def ratio(self, lst):
        odor,non_odor = 0,0
        for i in lst:
            if i<=0.5:
                non_odor+=1
            else:
                odor+=1
        try:
            ratio = odor/non_odor
        except ZeroDivisionError:
            if odor == 0:
                ratio = 1
            else:
                ratio = odor
        return float(ratio)

    def create_graph(self, name):
        graph_dir = self.graph_dir
        df = pd.read_csv(f'{graph_dir}/ratio_vs_time.csv')
        print(df)
        ax = sns.lineplot(x="time", y="ratio",
                        data=df)

        ax.set(xlabel='time', ylabel='ratio of number of odor and non odor')
        plt.xlabel("time")
        plt.ylabel("ratio of number of odor and non odor")
        plt.savefig(f"{graph_dir}/{name}.png")

    def update_ratio_file(self, time, ratio):
        df = pd.DataFrame({
            "time" : [time],
            'ratio' : [ratio]
        })
        df.to_csv(f'{self.graph_dir}/ratio_vs_time.csv', mode='a', header=False, index=False)
        self.create_graph(time)

    def str_to_float(self,i):
        if i == 'error':
            return float(0)
        else:
            return float(i)
    # this function
    def check_m1(self, name):
        input_path = f'{self.generate_dir}/{name}.csv'
        result_path = f'{self.m1_result_dir}/{name}.csv'
        os.system(f"python {self.m1_dir}/{self.odor_model} '{input_path}' '{result_path}' '{self.m1_dir}'")
        probs = pd.read_csv(f'{self.m1_result_dir}/{name}.csv')
        # for i in range(len(probs)):
        #     print(probs.iloc[i])
        lst = list(map(self.str_to_float, probs['property']))
        ratio = self.ratio(lst)
        self.update_ratio_file(name, ratio)

    def clear_data(self):
        os.system(f"rm {self.generate_dir}/*")
        os.system(f"rm {self.m1_result_dir}/*")
        dir = self.graph_dir
        for f in os.listdir(dir):
            if f.endswith(".csv"):
                continue
            os.remove(os.path.join(dir, f))
        df = pd.DataFrame({
            "time" : [],
            'ratio' : []
        })
        df.to_csv(f"{self.graph_dir}/ratio_vs_time.csv",index=False)

    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            log = ""
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch(self.test_batch_size)
            z = torch.tensor(self.sample_z(a.shape[0])).type(torch.FloatTensor)
            z = z.to(self.device)
            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = - torch.mean(logits_fake)

            # Fake Reward
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]

            # Log update
            # 'mols' is output of Fake Reward
            m0, m1 = all_scores(mols, self.m1_output, self.m1_input, self.data, norm=True)
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            for tag, value in m0.items():
                log += ", {}: {:.4f}".format(tag, value)

        name = self.generate_smiles(mols,self.generate_dir)
        # self.check_m1(name)
