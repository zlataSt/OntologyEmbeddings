# -*- coding: utf-8 -*-
import dill
from click.core import F
from torch.nn.init import xavier_uniform_
from torchkge import MarginLoss
from torchkge.models.bilinear import DistMultModel, RESCALModel, HolEModel, ComplExModel
from torchkge.models.translation import TransEModel

from embeddings.Evaluator import *
from embeddings.Loss import *


class JOIE:
    def __init__(self, kg_instance, kg_ontology, kg_ins_type,
                 entity_dim, concept_dim,
                 intra_view_model, cross_view_model,
                 margin_intra, margin_cross, **kwargs):

        # размерность эмбеддингов
        self.entity_dim = entity_dim
        self.concept_dim = concept_dim
        # параметры функции потерь
        self.margin_intra = margin_intra
        self.margin_cross = margin_cross
        # графы, из которых мы извлечем кол-во enteties,concpets,relations
        # + создадим нужную индексацию под созданные пространства эмбеддингов
        # self.kg = {'instance': kg_instance, 'ontology': kg_ontology, 'type': kg_ins_type}
        # типы моделей
        self.type_intra_models = intra_view_model
        self.type_cross_models = cross_view_model
        self.init_joie(kg_instance, kg_ontology, kg_ins_type, **kwargs)

    def kg_index_dicts(self, kg_instance, kg_ontology, kg_ins_type):

        '''
		метод, который собирает все enteties и concepts из графов, 
		потом собираем свои словари для проведения реиндексации во время обучения
		'''

        # соберем все enteties из kg_instance+kg_ins_type
        entities_from_instances = list(kg_instance.ent2ix.keys())
        entities_from_instype = kg_ins_type.get_df()['from'].unique().tolist()
        enteties = sorted(set(entities_from_instances) | set(entities_from_instype))
        # соберем все concepts из kg_ontology+kg_ins_type
        concepts_from_instances = list(kg_ontology.ent2ix.keys())
        concepts_from_instype = kg_ins_type.get_df()['to'].unique().tolist()
        concepts = sorted(set(concepts_from_instances) | set(concepts_from_instype))

        # узнаем количество enteties и concepts
        self.n_ent = len(enteties)
        self.n_con = len(concepts)
        self.n_rel_ent = kg_instance.n_rel
        self.n_rel_con = kg_ontology.n_rel
        # соберем общий словарь, где будет храниться и entety и concept со своим новым индексом
        self.example2idx = dict(zip(enteties, range(self.n_ent)))
        self.example2idx.update(dict(zip(concepts, range(self.n_con))))

    def init_embeddings(self):
        # инициализируем для concept+enteties эмбеддинги
        self.entetity_embedding = nn.Embedding(self.n_ent, self.entity_dim)
        xavier_uniform_(self.entetity_embedding.weight.data)

        self.concept_embedding = nn.Embedding(self.n_con, self.concept_dim)
        xavier_uniform_(self.concept_embedding.weight.data)

    def init_models(self, lr_cross=0.1, lr_instance=0.02, lr_ontology=0.01):

        # создаем модель, пока так просто 
        intra_models = {
            'transe': TransEModel, 'distmult': DistMultModel,
            'hole': HolEModel, 'complex': ComplExModel,
            'rescal': RESCALModel
        }

        intra_view_instance_model = intra_models[self.type_intra_models.lower()](emb_dim=self.entity_dim,
                                                                                 n_entities=self.n_ent,
                                                                                 n_relations=self.n_rel_ent)
        intra_view_ontology_model = intra_models[self.type_intra_models.lower()](emb_dim=self.concept_dim,
                                                                                 n_entities=self.n_con,
                                                                                 n_relations=self.n_rel_con)
        # создаем cross модель
        if self.type_cross_models.lower() == 'cg':
            cross_model = CrossViewGroupping(n_entities=self.n_ent, n_concepts=self.n_con,
                                             entity_dim=self.entity_dim, concept_dim=self.entity_dim, p=2)
            loss_cross = MarginLoss_CVG(self.margin_cross)

        elif self.type_cross_models.lower() == 'ct':
            cross_model = CrossViewTransformation(n_entities=self.n_ent, n_concepts=self.n_con,
                                                  entity_dim=self.entity_dim, concept_dim=self.concept_dim, p=2)
            loss_cross = MarginLoss_CVT(self.margin_cross)

        # меняем в моделях эмбеддинги чтобы они были общие

        cross_model.ent_embeddings = self.entetity_embedding
        cross_model.concept_embeddings = self.concept_embedding

        intra_view_instance_model.ent_emb = self.entetity_embedding
        intra_view_ontology_model.ent_emb = self.concept_embedding

        self.models = {
            'instance': intra_view_instance_model,
            'ontology': intra_view_ontology_model,
            'type': cross_model
        }

        optimizer_intra_instance = torch.optim.Adam(intra_view_instance_model.parameters(), lr=lr_instance,
                                                    amsgrad=True)
        optimizer_intra_ontology = torch.optim.Adam(intra_view_ontology_model.parameters(), lr=lr_ontology,
                                                    amsgrad=True)
        # Что здесь происходит ОЛЬГА
        cross_model.concept_embeddings.weight.requires_grad = False
        optimizer_cross = torch.optim.Adam(filter(lambda p: p.requires_grad, cross_model.parameters()), lr=lr_cross,
                                           amsgrad=True)
        cross_model.concept_embeddings.weight.requires_grad = True

        self.optimizers = {
            'instance': optimizer_intra_instance,
            'ontology': optimizer_intra_ontology,
            'type': optimizer_cross
        }
        self.criterions = {
            'instance': MarginLoss(self.margin_intra),
            'ontology': MarginLoss(self.margin_intra),
            'type': loss_cross
        }

    def init_joie(self, kg_instance, kg_ontology, kg_ins_type,
                  lr_cross=0.0001, lr_instance=0.0005, lr_ontology=0.001):

        self.kg_index_dicts(kg_instance, kg_ontology, kg_ins_type)
        self.init_embeddings()
        self.init_models(lr_cross, lr_instance, lr_ontology)

    def joie_performance(self, kg_instance, kg_ontology, kg_ins_type, b_size=64, candidates_b_size=100):
        '''оценка модели на переданных графах'''
        performance = {}
        instance_evaluation = self.evaluation(kg_instance, 'instance', b_size, candidates_b_size)

        performance['instance'] = {

            'mrr': instance_evaluation.mrr(),
            'hit_at_1': instance_evaluation.hit_at_k(1),
            'hit_at_3': instance_evaluation.hit_at_k(3),
            'hit_at_10': instance_evaluation.hit_at_k(10),
            'evaluator': instance_evaluation
        }

        ontology_evaluation = self.evaluation(kg_ontology, 'ontology', b_size, candidates_b_size)

        performance['ontology'] = {
            'mrr': ontology_evaluation.mrr(),
            'hit_at_1': ontology_evaluation.hit_at_k(1),
            'hit_at_3': ontology_evaluation.hit_at_k(3),
            'hit_at_10': ontology_evaluation.hit_at_k(10),
            'evaluator': ontology_evaluation
        }

        type_evaluation = self.evaluation(kg_ins_type, 'type', b_size)

        performance['type'] = {
            'mrr': type_evaluation.mrr(),
            'hit_at_1': type_evaluation.hit_at_k(1),
            'hit_at_3': type_evaluation.hit_at_k(3),
            'evaluator': type_evaluation
        }

        return performance

    def evaluation(self, kg, type_kg, b_size=64, candidates_b_size=100, max_better=False):
        ''' метод для подготовительных расчетов для метрик'''
        if type_kg == 'type':
            evaluator = CrossEvaluator(self.models[type_kg], kg, self.example2idx)
            evaluator.evaluate(b_size, max_better)
            return evaluator
        elif type_kg in ['instance', 'ontology']:
            evaluator = JoieLinkPredictionEvaluator(self, kg, type_kg)
            evaluator.evaluate(b_size, candidates_b_size)
            return evaluator

    def dump_model(self, filename):
        with open(filename, 'wb') as joie_model:
            dill.dump(self, joie_model)


class BaseCrossViewModel(nn.Module):
    """Description.
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    """

    def __init__(self, n_entities, n_concepts, entity_dim, concept_dim, p=2):
        super().__init__()
        self.n_entities = n_entities
        self.n_concepts = n_concepts

        self.entity_dim = entity_dim
        self.concept_dim = concept_dim
        self.p = p

        self.ent_embeddings = nn.Embedding(n_entities, self.entity_dim)
        xavier_uniform_(self.ent_embeddings.weight.data)

        self.concept_embeddings = nn.Embedding(n_concepts, self.concept_dim)
        xavier_uniform_(self.concept_embeddings.weight.data)

        self.rel_embeddings = nn.Embedding(1, self.entity_dim)
        xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self, heads, tails, relations, negative_heads, negative_tails, negative_relations=None):
        """
        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads --> entities for cross view models
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails. concepts for cross view models
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations --> meta relation.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.
        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.
        """
        pos = self.scoring_function(heads, tails, relations)
        neg = self.scoring_function(heads, negative_tails, relations)

        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.
        Parameters
        ----------
        h_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's heads
        t_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's tails.
        r_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's relations.
        Returns
        -------
        score: torch.Tensor, dtype: torch.float, shape: (b_size)
            Score of each triplet.
        """
        raise NotImplementedError

    def get_embeddings(self):
        """Return the tensors representing entities and relations in current
        model.
        """
        return self.ent_embeddings.weight.data, self.concept_embeddings.weight.data, self.rel_embeddings.weight.data

    def normalize_parameters(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=self.p, dim=-1)
        self.concept_embeddings.weight.data = F.normalize(self.concept_embeddings.weight.data, p=self.p, dim=-1)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.
        """
        b_size = h_idx.shape[0]

        h = self.ent_embeddings(h_idx)
        t = self.concept_embeddings(t_idx)
        r = self.rel_embeddings(r_idx)

        candidates = self.concept_embeddings.weight.data.view(self.n_concepts, self.concept_dim)
        candidates = candidates.expand(b_size, self.n_concepts, self.concept_dim)

        return h, t, r, candidates

    def loss(self, h, t):
        '''
        Function to calculate loss for cross view_model
        '''
        return torch.linalg.norm(t - h, dim=-1, ord=self.p)


class CrossViewGroupping(BaseCrossViewModel):
    """Description.
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    """

    def __init__(self, n_entities, n_concepts, entity_dim, concept_dim, p=2):
        super().__init__(n_entities, n_concepts, entity_dim, concept_dim, p)

    def scoring_function(self, h_idx, t_idx, r_idx):
        h, t, r, _ = self.inference_prepare_candidates(h_idx, t_idx, r_idx)
        return self.loss(h, t).flatten()

    def inference_scoring_function(self, proj_h, proj_t, r):
        """
        This method uses in torchkge.evaluation.LinkPredictionEvaluator to compute metrics
        """
        b_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                # this is the tail completion case in link prediction
                h_ = (proj_h).view(b_size, 1, proj_h.shape[1])
                return self.loss(h_, proj_t)


class CrossViewTransformation(BaseCrossViewModel):
    """Description.
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    """

    def __init__(self, n_entities, n_concepts, entity_dim, concept_dim, p=2):
        super().__init__(n_entities, n_concepts, entity_dim, concept_dim, p)
        self.linear = nn.Linear(entity_dim, concept_dim)
        self.tanh = nn.Tanh()
        # self.tanh.requires_grad_(False)
        # self.linear.requires_grad_(False)

    def linear_transformation(self, h):
        return self.tanh(self.linear(h))

    def scoring_function(self, h_idx, t_idx, r_idx):
        h, t, r, _ = self.inference_prepare_candidates(h_idx, t_idx, r_idx)

        return self.loss(self.linear_transformation(h), t).flatten()

    def inference_scoring_function(self, proj_h, proj_t, r):
        """
        This method uses in torchkge.evaluation.LinkPredictionEvaluator to compute metrics
        """
        b_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                # this is the tail completion case in link prediction
                h_transform = self.linear_transformation(proj_h)
                h_ = (h_transform).view(b_size, 1, proj_h.shape[1])
                return self.loss(h_, proj_t)

        # %%

class BaseEmbeddingModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(BaseEmbeddingModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, s, p, o):
        s_emb = self.entity_embeddings(s)
        p_emb = self.relation_embeddings(p)
        o_emb = self.entity_embeddings(o)
        score = torch.sum(s_emb * p_emb * o_emb, dim=1)
        return score
