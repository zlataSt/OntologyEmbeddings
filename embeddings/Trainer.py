from typing import Optional

import matplotlib.pyplot as plt
from torchkge.data_structures import SmallKG
from torchkge.sampling import BernoulliNegativeSampler, UniformNegativeSampler
from torchkge.utils.data import get_n_batches
from tqdm.autonotebook import tqdm

from embeddings.NegativeOntoSampler import *
from embeddings.Utils import *


class TrainDataLoader:

    def __init__(self, kg, batch_size, sampling_type, use_cuda=None):
        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.use_cuda = use_cuda
        self.b_size = batch_size
        self.iterator = None

        if sampling_type == 'unif':
            self.sampler = UniformNegativeSampler(kg)
        elif sampling_type == 'bern':
            self.sampler = BernoulliNegativeSampler(kg)
        else:
            self.sampler = sampling_type

        self.tmp_cuda = use_cuda in ['batch', 'all']

    def __len__(self):
        return get_n_batches(len(self.h), self.b_size)

    def __iter__(self):
        self.iterator = TrainDataLoaderIter(self)
        return self.iterator

    def get_counter_examples(self) -> SmallKG:
        return SmallKG(self.iterator.nh, self.iterator.nt, self.iterator.r)


class TrainDataLoaderIter:
    def __init__(self, loader):
        self.h = loader.h
        self.t = loader.t
        self.r = loader.r

        self.nh, self.nt = loader.sampler.corrupt_kg(loader.b_size, None)

        self.use_cuda = loader.tmp_cuda
        self.b_size = loader.b_size

        self.n_batches = get_n_batches(len(self.h), self.b_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            batch = dict()
            batch['h'] = self.h[i * self.b_size: (i + 1) * self.b_size]
            batch['t'] = self.t[i * self.b_size: (i + 1) * self.b_size]
            batch['r'] = self.r[i * self.b_size: (i + 1) * self.b_size]
            batch['nh'] = self.nh[i * self.b_size: (i + 1) * self.b_size]
            batch['nt'] = self.nt[i * self.b_size: (i + 1) * self.b_size]

            if self.use_cuda:
                batch['h'] = batch['h'].cuda()
                batch['t'] = batch['t'].cuda()
                batch['r'] = batch['r'].cuda()
                batch['nh'] = batch['nh'].cuda()
                batch['nt'] = batch['nt'].cuda()

            return batch

    def __iter__(self):
        return self


class Trainer:

    def __init__(self, model, criterion, kg_train, n_epochs, batch_size,
                 optimizer, sampling_type='bern', use_cuda=None):

        self.model = model
        self.criterion = criterion
        self.kg_train = kg_train
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.sampling_type = sampling_type

        self.batch_size = batch_size
        self.n_triples = len(kg_train)
        self.counter_examples: Optional[SmallKG] = None

    def process_batch(self, current_batch):
        self.optimizer.zero_grad()

        h, t, r = current_batch['h'], current_batch['t'], current_batch['r']
        nh, nt = current_batch['nh'], current_batch['nt']

        p, n = self.model(h, t, r, nh, nt)
        loss = self.criterion(p, n)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def run(self):
        if self.use_cuda in ['all', 'batch']:
            self.model.cuda()
            self.criterion.cuda()

        iterator = tqdm(range(self.n_epochs), unit='epoch')
        data_loader = TrainDataLoader(self.kg_train,
                                      batch_size=self.batch_size,
                                      sampling_type=self.sampling_type,
                                      use_cuda=self.use_cuda)

        for epoch in iterator:
            sum_ = 0
            for i, batch in enumerate(data_loader):
                loss = self.process_batch(batch)
                sum_ += loss

            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, sum_ / len(data_loader)))
            self.model.normalize_parameters()
        self.counter_examples = data_loader.get_counter_examples()

    def get_counter_examples(self):
        return self.counter_examples


class TrainDataLoaderJoie(TrainDataLoader):

    def __init__(self, kg_instance, kg_ontology, kg_ins_type, example2idx, n_batches, sampling_type, use_cuda=None):
        # исходя из словаря example2idx создадим свои head_idx,tail_idx, чтобы они однозначно
        self.example2idx = example2idx
        # что здесь происходит ОЛЯЯЯЯЯЯЯ???

        instance_head_idx, instance_tail_idx = (
            reindex_kg_idx_enteties_vector(kg_instance.head_idx, kg_instance, example2idx),
            reindex_kg_idx_enteties_vector(kg_instance.tail_idx, kg_instance, example2idx))
        ontology_head_idx, ontology_tail_idx = (
            reindex_kg_idx_enteties_vector(kg_ontology.head_idx, kg_ontology, example2idx),
            reindex_kg_idx_enteties_vector(kg_ontology.tail_idx, kg_ontology, example2idx))
        ins_type_head_idx, ins_type_tail_idx = (
            reindex_kg_idx_enteties_vector(kg_ins_type.head_idx, kg_ins_type, example2idx),
            reindex_kg_idx_enteties_vector(kg_ins_type.tail_idx, kg_ins_type, example2idx))

        self.help_dict = {
            'instance':
                {'positive': {'h': instance_head_idx,
                              't': instance_tail_idx, 'r': kg_instance.relations}},
            'ontology':
                {'positive': {'h': ontology_head_idx,
                              't': ontology_tail_idx, 'r': kg_ontology.relations}},
            'type':
                {'positive': {'h': ins_type_head_idx,
                              't': ins_type_tail_idx, 'r': kg_ins_type.relations}}
        }

        self.n_batches = n_batches

        self.iterator = None

        if sampling_type == 'unif':
            self.sampler_instance = UniformNegativeSampler(kg_instance)
            self.sampler_ontology = UniformNegativeSampler(kg_ontology)
        elif sampling_type == 'bern':
            self.sampler_instance = BernoulliNegativeSampler(kg_instance)
            self.sampler_ontology = BernoulliNegativeSampler(kg_ontology)

        self.sampler_instype = CrossNegativeSampler(kg_ins_type)

        self.use_cuda = use_cuda in ['batch', 'all', True, 1]

    def __len__(self):
        return self.n_batches

    def batch_sizes(self):

        return {'instance': len(self.help_dict['instance']['positive']['h']) // self.n_batches + 1,
                'ontology': len(self.help_dict['ontology']['positive']['h']) // self.n_batches + 1,
                'type': len(self.help_dict['type']['positive']['h']) // self.n_batches + 1}

    def __iter__(self):
        self.iterator = TrainDataLoaderIterJoie(self)
        return self.iterator

    def get_counter_examples(self) -> SmallKG:

        return {'instance': SmallKG(self.iterator.help_dict['instance']['negative']['h'],
                                    self.iterator.help_dict['instance']['negative']['t'],
                                    self.iterator.help_dict['instance']['positive']['r']),
                'ontology': SmallKG(self.iterator.help_dict['ontology']['negative']['h'],
                                    self.iterator.help_dict['ontology']['negative']['t'],
                                    self.iterator.help_dict['ontology']['positive']['r']),
                'type': SmallKG(self.iterator.help_dict['type']['negative']['h'],
                                self.iterator.help_dict['type']['negative']['t'],
                                self.iterator.help_dict['type']['positive']['r'])}


class TrainDataLoaderIterJoie(TrainDataLoaderIter):
    def __init__(self, loader):
        self.help_dict = loader.help_dict

        # что здесь происходит ОЛЯЯЯЯЯЯЯ???
        samplers = {'instance': loader.sampler_instance,
                    'ontology': loader.sampler_ontology, 'type': loader.sampler_instype}
        for type_kg in ['instance', 'ontology', 'type']:
            nh, nt = samplers[type_kg].corrupt_kg(
                loader.batch_sizes()[type_kg], None)

            nh_reindex = reindex_kg_idx_enteties_vector(
                nh, samplers[type_kg].kg, loader.example2idx)
            nt_reindex = reindex_kg_idx_enteties_vector(
                nt, samplers[type_kg].kg, loader.example2idx)

            self.help_dict[type_kg]['negative'] = dict(
                zip(['h', 't'], (nh_reindex, nt_reindex)))

        self.use_cuda = loader.use_cuda
        self.b_sizes = loader.batch_sizes()

        self.n_batches = loader.n_batches
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1
            # что здесь происходит ОЛЯЯЯЯЯЯЯ???
            big_batch = {}
            for type_kg in self.help_dict.keys():
                batch = dict()
                tb = self.b_sizes[type_kg]
                for type_sampler in self.help_dict[type_kg].keys():
                    for o in self.help_dict[type_kg][type_sampler].keys():
                        if type_sampler == 'positive':
                            batch[o] = self.help_dict[type_kg][type_sampler][o][i * tb: (i + 1) * tb]
                        else:
                            batch[f'n{o}'] = self.help_dict[type_kg][type_sampler][o][i * tb: (i + 1) * tb]

                if self.use_cuda:
                    for key in batch.keys():
                        batch[key] = batch[key].cuda()

                big_batch[type_kg] = batch

            return big_batch

    def __iter__(self):
        return self


class TrainerJoie(Trainer):
    def __init__(self, joie,
                 kg_instance, kg_ontology, kg_instype,
                 n_epochs, n_batches,
                 alpha=1, omega=1,
                 sampling_type='bern', use_cuda=None):

        self.joie = joie

        self.kg_train_instances = kg_instance
        self.kg_train_ontology = kg_ontology
        self.kg_train_type = kg_instype

        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.sampling_type = sampling_type

        self.n_batches = n_batches

        self.counter_examples: Optional[SmallKG] = None
        self.coefs = {'instance': 1, 'ontology': alpha, 'type': omega}

    def process_batch(self, current_batch, type_kg):
        self.joie.optimizers[type_kg].zero_grad()
        # что здесь происходит ОЛЯЯЯЯЯЯЯ???
        h, t, r = current_batch[type_kg]['h'], current_batch[type_kg]['t'], current_batch[type_kg]['r']
        nh, nt = current_batch[type_kg]['nh'], current_batch[type_kg]['nt']
        # что здесь происходит ОЛЯЯЯЯЯЯЯ???
        p, n = self.joie.models[type_kg](h, t, r, nh, nt)
        loss = self.coefs[type_kg] * self.joie.criterions[type_kg](p, n)
        # что здесь происходит ОЛЯЯЯЯЯЯЯ???
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.joie.models[type_kg].parameters(), 1)
        self.joie.optimizers[type_kg].step()

        return loss.detach().item()

    def train_epoch(self, dataloader, type_kg):
        # выключим по теоретическим соображениям cross view модели обучения эмбеддингов для концептов
        loss = 0
        step = 0
        for i, batch in enumerate(dataloader):
            loss += self.process_batch(batch, type_kg)
            step += 1

        return loss / step

    def run(self, off_cross_gap=0, eps=0.5):
        if self.use_cuda in ['all', 'batch', True, 1]:
            for type_kg in ['instance', 'ontology', 'type']:
                self.joie.models[type_kg].cuda()
                self.joie.criterions[type_kg].cuda()

        off_cross = int(off_cross_gap) != 0

        iterator = tqdm(range(self.n_epochs), unit='epoch')
        self.dataloader = TrainDataLoaderJoie(
            kg_instance=self.kg_train_instances,
            kg_ontology=self.kg_train_ontology,
            kg_ins_type=self.kg_train_type,
            example2idx=self.joie.example2idx,
            n_batches=self.n_batches,
            sampling_type=self.sampling_type,
            use_cuda=self.use_cuda)
        self.train_history = {'instance': [-1000], 'ontology': [-1000], 'type': [-1000], 'joie_loss': [-1000]}
        pass_epoch = 0

        for epoch in iterator:
            # что здесь происходит ОЛЯЯЯЯЯЯЯ???
            losses = {'instance': 0, 'ontology': 0, 'type': 0}
            joie_loss = 0
            for type_kg in ['instance', 'ontology', 'type']:
                # ОЛЯЯЯЯЯЯ что здесь происходит???
                # для остановки обучения кросс модели
                if off_cross and type_kg == 'type' and abs(self.train_history[type_kg][-1]) <= off_cross_gap:
                    losses[type_kg] = self.train_history[type_kg][-1]
                    self.train_history[type_kg].append(self.train_history[type_kg][-1])
                    joie_loss += self.train_history[type_kg][-1]
                    pass_epoch += 1
                    continue

                elif type_kg == 'type' and pass_epoch >= 10:
                    pass_epoch = 0

                losses[type_kg] = self.train_epoch(self.dataloader, type_kg)
                self.train_history[type_kg].append(losses[type_kg])
                joie_loss += losses[type_kg]

            self.train_history['joie_loss'].append(joie_loss)
            iterator.set_description(
                'Epoch {} | Average losses: embeddings loss: {:.5f} | loss intra view intstance model: {:.5f} | loss intra ontology model: {:.5f} | loss cross view model: {:.5f}'.format(
                    epoch + 1, joie_loss, losses['instance'], losses['ontology'], losses['type']))

            self.joie.models['instance'].normalize_parameters()
            self.joie.models['ontology'].normalize_parameters()

        self.train_history['instance'].pop(0)
        self.train_history['ontology'].pop(0)
        self.train_history['type'].pop(0)
        self.train_history['joie_loss'].pop(0)

        self.counter_examples = self.dataloader.get_counter_examples()

    def get_counter_examples(self) -> Optional[SmallKG]:
        return self.counter_examples

    def plot_train_process(self):
        type_kg = ['Intra view instance model loss', 'Intra view ontology model loss', 'Cross view type model loss']
        fig, axes = plt.subplots(1, 3, figsize=(30, 6))
        for i in range(3):
            kg = type_kg[i].split()[2]
            x = self.train_history[kg]
            axes[i].plot(list(range(len(x))), x)
            axes[i].grid()
            axes[i].set_xlabel('epoch')
            axes[i].set_ylabel('margin loss')
            axes[i].set_title(type_kg[i])
