# %%
import os
import traceback
import random
import pickle
import numpy as np
import rdkit
from rdkit import Chem
import tensorflow as tf
import pickle
import gzip
import random
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import pandas as pd
import math
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from collections import defaultdict
import pprint


# %%
# layers.py

def graph_convolution_layer(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))

    annotations = tf.concat(
        (hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor

    output = tf.stack([tf.layers.dense(inputs=annotations, units=units)
                      for _ in range(adj.shape[1])], 1)

    output = tf.matmul(adj, output)
    output = tf.reduce_sum(output, 1) + \
        tf.layers.dense(inputs=annotations, units=units)
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def graph_aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid)
    j = tf.layers.dense(inputs, units=units, activation=activation)
    output = tf.reduce_sum(i * j, 1)
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs
    for u in units:
        hidden_tensor = tf.layers.dense(
            hidden_tensor, units=u, activation=activation)
        hidden_tensor = tf.layers.dropout(
            hidden_tensor, dropout_rate, training=training)

    return hidden_tensor


def multi_graph_convolution_layers(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = graph_convolution_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                                units=u, activation=activation, dropout_rate=dropout_rate,
                                                training=training)
    return hidden_tensor


# %%
# molecular_metrics.py
NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(
    gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}


class MolecularMetrics(object):
    odor_result = None

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(
            x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols, norm=False):
        # calculating the score
        scores = [sum(NP_model.get(bit, 0)for bit in Chem.rdMolDescriptors.GetMorganFingerprint(
            mol, 2).GetNonzeroElements()) / float(mol.GetNumAtoms()) if mol is not None else None for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (4 + math.log10(score - 4 + 1)
                      if score > 4 else (-4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(
            scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(
            scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0

        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        print("getnumatoms", nAtoms)
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
            spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(
            mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(
            scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(
            ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):
        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(
                mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4
        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))

    # @staticmethod
    # def odor_compute(smile):
    #     if odor_result[smile] < 0.5:
    #         return False
    #     return True

    @staticmethod
    def odor_score(mols):
        one = 0
        total = 0
        for i in range(len(mols)):
            if MolecularMetrics.odor_result[i] > 0.5:
                one += 1
            total += 1
        return float(one/total)

    @staticmethod
    def write_result(mols, smiles):
        f = open("logs.txt", 'a')
        f.write("\n")
        for i in range(len(mols)):
            f.write(
                f"{smiles[i]}   {round(MolecularMetrics.odor_result[i],4)}\n")
        f.write(f"Accuracy - {MolecularMetrics.odor_score(mols)}\n")

    @staticmethod
    def odor(mols):
        smiles = []
        temp1 = []
        temp2 = []
        count = 0
        for i in mols:
            if i is not None:
                temp1.append(count)
                temp = Chem.MolToSmiles(i)
                smiles.append(temp)
                temp2.append(temp)
                count += 1
            else:
                temp2.append(None)

        result = [0 for i in range(len(mols))]
        dic = {"smiles": smiles}
        df = pd.DataFrame(dic)
        os.chdir("m1")
        df.to_csv(f"input.csv", index=False)
        !python transformer-cnn.py

        try:
            results = pd.read_csv(f"results.csv")
            temp = []
            for i in results['property']:
                if i == 'error':
                    temp.append(0)
                else:
                    temp.append(float(i))
            results = temp
            count = 0
            for i in temp1:
                result[i] = results[count]
                count += 1
            os.chdir("..")
            MolecularMetrics.odor_result = result
            MolecularMetrics.write_result(mols, temp2)
            return np.array(result)*0.1
        except:
            os.chdir("..")
            traceback.print_exc()
            exit()

# %%
# progressbar.py


class ProgressBar:
    def __init__(self, length, max_value):
        assert length > 0 and max_value > 0
        self.length, self.max_value, self.start = length, max_value, time.time()

    def update(self, value):
        print(value, "progress bar")
        #assert 0 < value <= self.max_value
        delta = (time.time() - self.start) * (self.max_value - value) / value
        format_spec = [value / self.max_value,
                       value,
                       len(str(self.max_value)),
                       self.max_value,
                       len(str(self.max_value)),
                       '#' * int((self.length * value) / self.max_value),
                       self.length,
                       timedelta(seconds=int(delta))
                       if delta < 60 * 60 * 10 else '-:--:-']
        #print('\r{:=5.0%} ({:={}}/{:={}}) [{:{}}] ETA: {}'.format(*format_spec), end='')

# %%
# sparse molecular


class SparseMolecularDataset():
    def load(self, filename, subset=1):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(
            len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(
            len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            self.data = list(filter(lambda x: x is not None,
                             Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            self.data = [Chem.MolFromSmiles(
                line) for line in open(filename, 'r').readlines()]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(
            len(self.data), len(Chem.SDMolSupplier(filename)), '' if add_h else 'not '))

        self._generate_encoders_decoders()
        self._generate_AX()

        # it contains the all the molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)

        # it contains the all the molecules stored as SMILES strings
        self.smiles = np.array(self.smiles)

        # a (N, L) matrix where N is the length of the dataset and each L-dim vector contains the
        # indices corresponding to a SMILE sequences with padding wrt the max length of the longest
        # SMILES sequence in the dataset (see self._genS)
        self.data_S = np.stack(self.data_S)

        # a (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        # (see self._genA)
        self.data_A = np.stack(self.data_A)

        # a (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        # (see self._genX)
        self.data_X = np.stack(self.data_X)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the
        # diagonal of the correspondent adjacency matrix
        self.data_D = np.stack(self.data_D)

        # a (N, F) matrix where N is the length of the dataset and each F vector contains features
        # of the correspondent molecule (see self._genF)
        self.data_F = np.stack(self.data_F)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the
        # eigenvalues of the correspondent Laplacian matrix
        self.data_Le = np.stack(self.data_Le)

        # a (N, 9, 9) matrix where N is the length of the dataset and each  9x9 matrix contains the
        # eigenvectors of the correspondent Laplacian matrix
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(
            set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(
            sorted(set(bond.GetBondType()for mol in self.data for bond in mol.GetBonds())))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = [
            'E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))

    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')
        print("len", len(self.data))
        pr = ProgressBar(len(self.data), 60)

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = np.diag(D) - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

                # print(i)
                pr.update(i+1)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(
            len(data), len(self.data)))

        self.data = data
        self.smiles = smiles
        self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [
            b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()]
                     for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
            max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(
            Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
            max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                              *[a.GetExplicitValence() == i for i in range(9)],
                              *[int(a.GetHybridization()) ==
                                i for i in range(1, 7)],
                              *[a.GetImplicitValence() == i for i in range(9)
                                ], a.GetIsAromatic(), a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)],
                              *[a.GetNumImplicitHs() == i for i in range(5)],
                              *[a.GetNumRadicalElectrons() == i for i in range(5)
                                ], a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2mol(self, node_labels, edge_labels, strict=False):

        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end),
                            self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):

        mol = Chem.MolFromSmiles(
            ''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X, self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A,
                                           self.data_X, self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output

    def next_train_batch(self, batch_size=None):

        out = self._next_batch(counter=self.train_counter,
                               count=self.train_count, idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter,
                               count=self.validation_count, idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(
            counter=self.test_counter, count=self.test_count, idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) +
              ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    data = SparseMolecularDataset()
    # data.generate('/content/drive/MyDrive/molgan/molgan/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('/content/drive/MyDrive/molgan/molgan/gdb9_9nodes.sparsedataset')

    # data = SparseMolecularDataset()
    # data.generate('data/qm9_5k.smi', validation=0.00021, test=0.00021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('data/qm9_5k.sparsedataset')

# %%
# utils.py


def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)])
                    for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol()
                    for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
          'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
          'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
          'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
          'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
          'diversity score': MolecularMetrics.diversity_scores(mols, data),
          'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100,
          #   'odor score': MolecularMetrics.odor_score(mols)
          }
    return m0, m1

# %%
# trainer.py


class Trainer:

    def __init__(self, model, optimizer, session):
        self.model, self.optimizer, self.session, self.print = model, optimizer, session, defaultdict(
            list)

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) +
              ' ' + str(msg) if date else str(msg))

    def save(self, directory):
        saver = tf.train.Saver()

        dirs = directory.split('/')
        dirs = ['/'.join(dirs[:i]) for i in range(1, len(dirs) + 1)]
        mkdirs = [d for d in dirs if not os.path.exists(d)]

        for d in mkdirs:
            os.makedirs(d)

        saver.save(self.session, '{}/{}.ckpt'.format(directory, 'model'))
        pickle.dump(self.print, open(
            '{}/{}.pkl'.format(directory, 'trainer'), 'wb'))
        self.log('Model saved in {}!'.format(directory))

    def load(self, directory):
        saver = tf.train.Saver()
        saver.restore(self.session, '{}/{}.ckpt'.format(directory, 'model'))
        self.print = pickle.load(
            open('{}/{}.pkl'.format(directory, 'trainer'), 'rb'))
        self.log('Model loaded from {}!'.format(directory))

    def train(self, batch_dim, epochs, steps,
              train_fetch_dict, train_feed_dict,
              eval_fetch_dict, eval_feed_dict,
              test_fetch_dict, test_feed_dict,
              _train_step=None, _eval_step=None, _test_step=None,
              _train_update=None, _eval_update=None, _test_update=None,
              eval_batch=None, test_batch=None,
              best_fn=None, min_epochs=None, look_ahead=None,
              save_every=None, directory=None,
              skip_first_eval=False):

        if _train_step is None:
            def _train_step(step, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
                return self.session.run(train_fetch_dict(step, steps, epoch, epochs, min_epochs, model, optimizer),
                                        feed_dict=train_feed_dict(step, steps, epoch, epochs, min_epochs, model,
                                                                  optimizer, batch_dim))

        if _eval_step is None:
            def _eval_step(epoch, epochs, min_epochs, model, optimizer, batch_dim, eval_batch, start_time,
                           last_epoch_start_time, _eval_update):
                from_start = timedelta(seconds=int((time.time() - start_time)))
                last_epoch = timedelta(seconds=int(
                    (time.time() - last_epoch_start_time)))
                eta = timedelta(seconds=int((time.time() - start_time) * (epochs - epoch) / epoch)
                                ) if (time.time() - start_time) > 1 else '-:--:-'

                self.log(
                    'Epochs {:10}/{} in {} (last epoch in {}), ETA: {}'.format(epoch, epochs, from_start, last_epoch,
                                                                               eta))

                if eval_batch is not None:
                    pr = ProgressBar(80, eval_batch)
                    output = defaultdict(list)

                    for i in range(eval_batch):
                        for k, v in self.session.run(eval_fetch_dict(epoch, epochs, min_epochs, model, optimizer),
                                                     feed_dict=eval_feed_dict(epoch, epochs, min_epochs, model,
                                                                              optimizer, batch_dim)).items():
                            output[k].append(v)
                        pr.update(i + 1)

                    self.log(date=False)
                    output = {k: np.mean(v) for k, v in output.items()}
                else:
                    output = self.session.run(eval_fetch_dict(epoch, epochs, min_epochs, model, optimizer),
                                              feed_dict=eval_feed_dict(epoch, epochs, min_epochs, model,
                                                                       optimizer, batch_dim))

                if _eval_update is not None:
                    output.update(_eval_update(
                        epoch, epochs, min_epochs, model, optimizer, batch_dim, eval_batch))

                p = pprint.PrettyPrinter(indent=1, width=80)
                self.log('Validation --> {}'.format(p.pformat(output)))

                for k in output:
                    self.print[k].append(output[k])

                return output

        if _test_step is None:

            def _test_step(model, optimizer, batch_dim, test_batch, start_time, _test_update):
                self.load(directory)
                from_start = timedelta(seconds=int((time.time() - start_time)))
                self.log('End of training ({} epochs) in {}'.format(
                    epochs, from_start))

                if test_batch is not None:
                    pr = ProgressBar(80, test_batch)
                    output = defaultdict(list)

                    for i in range(test_batch):
                        for k, v in self.session.run(test_fetch_dict(model, optimizer),
                                                     feed_dict=test_feed_dict(model, optimizer, batch_dim)).items():
                            output[k].append(v)
                        pr.update(i + 1)

                    self.log(date=False)
                    output = {k: np.mean(v) for k, v in output.items()}
                else:
                    output = self.session.run(test_fetch_dict(model, optimizer),
                                              feed_dict=test_feed_dict(model, optimizer, batch_dim))

                if _test_update is not None:
                    output.update(_test_update(
                        model, optimizer, batch_dim, test_batch))

                p = pprint.PrettyPrinter(indent=1, width=80)
                self.log('Test --> {}'.format(p.pformat(output)))

                for k in output:
                    self.print['Test ' + k].append(output[k])

                return output

        best_model_value = None
        no_improvements = 0
        start_time = time.time()
        last_epoch_start_time = time.time()

        for epoch in range(epochs + 1):
            if not (skip_first_eval and epoch == 0):
                result = _eval_step(epoch, epochs, min_epochs, self.model, self.optimizer, batch_dim, eval_batch,
                                    start_time, last_epoch_start_time, _eval_update)

                if best_fn is not None and (True if best_model_value is None else best_fn(result) > best_model_value):
                    self.save(directory)
                    best_model_value = best_fn(result)
                    no_improvements = 0
                elif look_ahead is not None and no_improvements < look_ahead:
                    no_improvements += 1
                    self.load(directory)
                elif min_epochs is not None and epoch >= min_epochs:
                    self.log('No improvements after {} epochs!'.format(
                        no_improvements))
                    break

                if save_every is not None and epoch % save_every == 0:
                    self.save(directory)

            if epoch < epochs:
                last_epoch_start_time = time.time()
                pr = ProgressBar(80, steps)
                for step in range(steps):
                    _train_step(steps * epoch + step, steps, epoch, epochs, min_epochs, self.model, self.optimizer,
                                batch_dim)
                    pr.update(step + 1)

                self.log(date=False)

        _test_step(self.model, self.optimizer, batch_dim,
                   eval_batch, start_time, _test_update)

# %%
# gan.py..optimiser


class GraphGANOptimizer(object):

    def __init__(self, model, learning_rate=1e-3, feature_matching=True):
        self.la = tf.placeholder_with_default(1., shape=())

        with tf.name_scope('losses'):
            eps = tf.random_uniform(tf.shape(model.logits_real)[
                                    :1], dtype=model.logits_real.dtype)

            x_int0 = model.adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1),
                                                             -1) + model.edges_softmax * (
                1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))
            x_int1 = model.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + model.nodes_softmax * (
                1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

            grad0, grad1 = tf.gradients(
                model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))

            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + tf.reduce_mean(
                ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)

            self.loss_D = - model.logits_real + model.logits_fake
            self.loss_G = - model.logits_fake
            self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + (
                model.value_logits_fake - model.rewardF) ** 2
            self.loss_RL = - model.value_logits_fake
            self.loss_F = (tf.reduce_mean(model.features_real, 0) -
                           tf.reduce_mean(model.features_fake, 0)) ** 2

        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(
            self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('train_step'):
            self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_D + 10 * self.grad_penalty,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) + tf.cond(
                    tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))
# %%
# vae.py..optimiser


class GraphVAEOptimizer(object):

    def __init__(self, model, learning_rate=1e-3):
        self.kl_weight = tf.placeholder_with_default(1., shape=())
        self.la = tf.placeholder_with_default(1., shape=())

        edges_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.edges_labels,
                                                            logits=model.edges_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.edges_loss = tf.reduce_sum(edges_loss, [-2, -1])

        nodes_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.nodes_labels,
                                                            logits=model.nodes_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.nodes_loss = tf.reduce_sum(nodes_loss, -1)

        self.loss_ = self.edges_loss + self.nodes_loss
        self.reconstruction_loss = tf.reduce_mean(self.loss_)

        self.p_z = tf.distributions.Normal(tf.zeros_like(model.embeddings_mean),
                                           tf.ones_like(model.embeddings_std))
        self.kl = tf.reduce_mean(tf.reduce_sum(
            tf.distributions.kl_divergence(model.q_z, self.p_z), axis=-1))

        self.ELBO = - self.reconstruction_loss - self.kl

        self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + \
            (model.value_logits_fake - model.rewardF) ** 2

        self.loss_RL = - model.value_logits_fake

        self.loss_RL = - model.value_logits_fake

        self.loss_VAE = tf.cond(model.variational,
                                lambda: self.reconstruction_loss + self.kl_weight * self.kl,
                                lambda: self.reconstruction_loss)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        self.loss_RL *= tf.abs(tf.stop_gradient(self.loss_VAE / self.loss_RL))

        self.VAE_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_VAE = self.VAE_optim.minimize(

            loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_VAE, lambda: 0.) + tf.cond(
                tf.less(self.la, 1), lambda: (1 - self.la) * self.loss_RL, lambda: 0.),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder'))

        self.V_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_V = self.V_optim.minimize(
            loss=self.loss_V,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))

        self.log_likelihood = self.__log_likelihood
        self.model = model

    def __log_likelihood(self, n):

        z = self.model.q_z.sample(n)

        log_p_z = self.p_z.log_prob(z)
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

        log_p_x_z = -self.loss_

        log_q_z_x = self.model.q_z.log_prob(z)
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

        print([a.shape for a in (log_p_z, log_p_x_z, log_q_z_x)])

        return tf.reduce_mean(tf.reduce_logsumexp(
            tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))

# %%
# models.....gan.py


class GraphGANModel(object):
    def __init__(self, vertexes, edges, nodes, embedding_dim, decoder_units, discriminator_units,
                 decoder, discriminator, soft_gumbel_softmax=False, hard_gumbel_softmax=False,
                 batch_discriminator=True):
        self.vertexes, self.edges, self.nodes, self.embedding_dim, self.decoder_units, self.discriminator_units, \
            self.decoder, self.discriminator, self.batch_discriminator = vertexes, edges, nodes, embedding_dim, decoder_units, \
            discriminator_units, decoder, discriminator, batch_discriminator

        self.training = tf.placeholder_with_default(False, shape=())
        self.dropout_rate = tf.placeholder_with_default(0., shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(
            soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(
            hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder_with_default(1., shape=())

        self.edges_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes, vertexes))
        self.nodes_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes))
        self.embeddings = tf.placeholder(
            dtype=tf.float32, shape=(None, embedding_dim))

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.adjacency_tensor = tf.one_hot(
            self.edges_labels, depth=edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(
            self.nodes_labels, depth=nodes, dtype=tf.float32)

        with tf.variable_scope('generator'):
            self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                                                                training=self.training, dropout_rate=self.dropout_rate)

        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
                (self.edges_argmax, self.nodes_argmax), \
                (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
                (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
                (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                (self.edges_logits, self.nodes_logits), temperature=self.temperature)

            self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                                     default=lambda: self.edges_softmax,
                                     exclusive=True)

            self.nodes_hat = tf.case({self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                                     default=lambda: self.nodes_softmax,
                                     exclusive=True)

        with tf.name_scope('D_x_real'):
            self.logits_real, self.features_real = self.D_x((self.adjacency_tensor, None, self.node_tensor),
                                                            units=discriminator_units)
        with tf.name_scope('D_x_fake'):
            self.logits_fake, self.features_fake = self.D_x((self.edges_hat, None, self.nodes_hat),
                                                            units=discriminator_units)

        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.V_x((self.adjacency_tensor, None, self.node_tensor),
                                              units=discriminator_units)
        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.V_x(
                (self.edges_hat, None, self.nodes_hat), units=discriminator_units)

    def D_x(self, inputs, units):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            outputs0 = self.discriminator(inputs, units=units[:-1], training=self.training,
                                          dropout_rate=self.dropout_rate)

            outputs1 = multi_dense_layers(outputs0, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                          dropout_rate=self.dropout_rate)

            if self.batch_discriminator:
                outputs_batch = tf.layers.dense(
                    outputs0, units[-2] // 8, activation=tf.tanh)
                outputs_batch = tf.layers.dense(tf.reduce_mean(outputs_batch, 0, keep_dims=True), units[-2] // 8,
                                                activation=tf.nn.tanh)
                outputs_batch = tf.tile(
                    outputs_batch, (tf.shape(outputs0)[0], 1))

                outputs1 = tf.concat((outputs1, outputs_batch), -1)

            outputs = tf.layers.dense(outputs1, units=1)

        return outputs, outputs1

    def V_x(self, inputs, units):
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.discriminator(inputs, units=units[:-1], training=self.training,
                                         dropout_rate=self.dropout_rate)

            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=self.dropout_rate)

            outputs = tf.layers.dense(
                outputs, units=1, activation=tf.nn.sigmoid)

        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))

# %%
# models...vae.py


#from models import postprocess_logits
#from utils.layers import multi_dense_layers


class GraphVAEModel:
    def __init__(self, vertexes, edges, nodes, features, embedding_dim, encoder_units, decoder_units, variational,
                 encoder, decoder, soft_gumbel_softmax=False, hard_gumbel_softmax=False, with_features=True):

        self.vertexes, self.nodes, self.edges, self.embedding_dim, self.encoder, self.decoder = \
            vertexes, nodes, edges, embedding_dim, encoder, decoder

        self.training = tf.placeholder_with_default(False, shape=())
        self.variational = tf.placeholder_with_default(variational, shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(
            soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(
            hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder_with_default(1., shape=())

        self.edges_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes, vertexes))
        self.nodes_labels = tf.placeholder(
            dtype=tf.int64, shape=(None, vertexes))
        self.node_features = tf.placeholder(
            dtype=tf.float32, shape=(None, vertexes, features))

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.adjacency_tensor = tf.one_hot(
            self.edges_labels, depth=edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(
            self.nodes_labels, depth=nodes, dtype=tf.float32)

        with tf.variable_scope('encoder'):
            outputs = self.encoder(
                (self.adjacency_tensor,
                 self.node_features if with_features else None, self.node_tensor),
                units=encoder_units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=encoder_units[-1], activation=tf.nn.tanh,
                                         training=self.training, dropout_rate=0.)

            self.embeddings_mean = tf.layers.dense(
                outputs, embedding_dim, activation=None)
            self.embeddings_std = tf.layers.dense(
                outputs, embedding_dim, activation=tf.nn.softplus)
            self.q_z = tf.distributions.Normal(
                self.embeddings_mean, self.embeddings_std)

            self.embeddings = tf.cond(self.variational,
                                      lambda: self.q_z.sample(),
                                      lambda: self.embeddings_mean)

        with tf.variable_scope('decoder'):
            self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                                                                training=self.training, dropout_rate=0.)

        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
                (self.edges_argmax, self.nodes_argmax), \
                (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
                (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
                (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                (self.edges_logits, self.nodes_logits), temperature=self.temperature)

            self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                                     default=lambda: self.edges_softmax,
                                     exclusive=True)

            self.nodes_hat = tf.case({self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                                     default=lambda: self.nodes_softmax,
                                     exclusive=True)

        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.V_x(
                (self.adjacency_tensor, None, self.node_tensor), units=encoder_units)

        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.V_x(
                (self.edges_hat, None, self.nodes_hat), units=encoder_units)

    def V_x(self, inputs, units):
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.encoder(
                inputs, units=units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=0.)

            outputs = tf.layers.dense(
                outputs, units=1, activation=tf.nn.sigmoid)

        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))

# %%
# models...init.py


#from utils.layers import multi_graph_convolution_layers, graph_aggregation_layer, multi_dense_layers


def encoder_rgcn(inputs, units, training, dropout_rate=0.):
    graph_convolution_units, auxiliary_units = units

    with tf.variable_scope('graph_convolutions'):
        output = multi_graph_convolution_layers(inputs, graph_convolution_units, activation=tf.nn.tanh,
                                                dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('graph_aggregation'):
        _, hidden_tensor, node_tensor = inputs
        annotations = tf.concat(
            (output, hidden_tensor, node_tensor) if hidden_tensor is not None else (output, node_tensor), -1)

        output = graph_aggregation_layer(annotations, auxiliary_units, activation=tf.nn.tanh,
                                         dropout_rate=dropout_rate, training=training)

    return output


def decoder_adj(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(
        inputs, units, activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(tf.layers.dense(inputs=output, units=edges * vertexes * vertexes,
                                                  activation=None), (-1, edges, vertexes, vertexes))
        edges_logits = tf.transpose(
            (edges_logits + tf.matrix_transpose(edges_logits)) / 2, (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(
            edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = tf.layers.dense(
            inputs=output, units=vertexes * nodes, activation=None)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
        nodes_logits = tf.layers.dropout(
            nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_dot(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(
        inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(tf.layers.dense(inputs=output, units=edges * vertexes * units[-1],
                                                  activation=None), (-1, edges, vertexes, units[-1]))
        edges_logits = tf.transpose(
            tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(
            edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = tf.layers.dense(
            inputs=output, units=vertexes * nodes, activation=None)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
        nodes_logits = tf.layers.dropout(
            nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_rnn(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(
        inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)

        edges_logits = tf.layers.dense(edges_logits, edges * units[-1])
        edges_logits = tf.transpose(tf.reshape(
            edges_logits, (-1, vertexes, edges, units[-1])), (0, 2, 1, 3))
        edges_logits = tf.transpose(
            tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(
            edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)
        nodes_logits = tf.layers.dense(nodes_logits, nodes)
        nodes_logits = tf.layers.dropout(
            nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def postprocess_logits(inputs, temperature=1.):

    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    softmax = [tf.nn.softmax(e_logits / temperature)
               for e_logits in listify(inputs)]
    argmax = [tf.one_hot(tf.argmax(e_logits, axis=-1), depth=e_logits.shape[-1], dtype=e_logits.dtype)
              for e_logits in listify(inputs)]
    gumbel_logits = [e_logits - tf.log(- tf.log(tf.random_uniform(tf.shape(e_logits), dtype=e_logits.dtype)))
                     for e_logits in listify(inputs)]
    gumbel_softmax = [tf.nn.softmax(e_gumbel_logits / temperature)
                      for e_gumbel_logits in gumbel_logits]
    gumbel_argmax = [
        tf.one_hot(tf.argmax(e_gumbel_logits, axis=-1),
                   depth=e_gumbel_logits.shape[-1], dtype=e_gumbel_logits.dtype)
        for e_gumbel_logits in gumbel_logits]

    return [delistify(e) for e in (softmax, argmax, gumbel_logits, gumbel_softmax, gumbel_argmax)]


# %%
# example.py
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

tf.reset_default_graph()


batch_dim = 128
la = 1
dropout = 0
n_critic = 5
metric = 'validity,unique'
n_samples = 5000
z_dim = 8
epochs = 10
save_every = 1  # May lead to errors if left as None

data = SparseMolecularDataset()
data.load('data/odor_smile_filtered.sparsedataset')

steps = (len(data) // batch_dim)


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    a = [optimizer.train_step_G] if i % n_critic == 0 else [
        optimizer.train_step_D]
    b = [optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
    return a + b


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    embeddings = model.sample_z(batch_dim)

    if la < 1:
        if i % n_critic == 0:
            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True)
                    for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: dropout,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def reward(mols):

    rr = 1.
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):
        if m == 'np':
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(
                mols, norm=True)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(
                mols, norm=True)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(
                mols, norm=True)
        elif m == 'novelty':
            rr *= MolecularMetrics.novel_scores(mols, data)
        elif m == 'dc':
            rr *= MolecularMetrics.drugcandidate_scores(mols, data)
        elif m == 'unique':
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= MolecularMetrics.diversity_scores(mols, data)
        elif m == 'validity':
            rr *= MolecularMetrics.valid_scores(mols)
        elif m == 'odor':
            rr *= MolecularMetrics.odor(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    mols = samples(data, model, session,
                   model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    mols = samples(data, model, session,
                   model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


# model
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((128, 64), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)


# optimizer
optimizer = GraphGANOptimizer(
    model, learning_rate=1e-3, feature_matching=False)

# session
session = tf.Session()
session.run(tf.global_variables_initializer())

# trainer
trainer = Trainer(model, optimizer, session)

print('Parameters: {}'.format(
    np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

trainer.train(batch_dim=batch_dim,
              epochs=epochs,
              steps=steps,
              train_fetch_dict=train_fetch_dict,
              train_feed_dict=train_feed_dict,
              eval_fetch_dict=eval_fetch_dict,
              eval_feed_dict=eval_feed_dict,
              test_fetch_dict=test_fetch_dict,
              test_feed_dict=test_feed_dict,
              save_every=save_every,
              # here users need to first create and then specify a folder where to save the model
              directory='example_model',
              _eval_update=_eval_update,
              _test_update=_test_update,
              )


# %%
