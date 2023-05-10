import gc

from torch import empty
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.utils import get_rank, filter_scores
from tqdm.autonotebook import tqdm

from embeddings.Utils import *


class NotYetEvaluatedError(Exception):
    def __init__(self, message):
        super(self, message).__init__()


class CrossEvaluator:
    def __init__(self, cross_model, kg_ins_type, example2idx):
        self.model = cross_model
        self.kg = kg_ins_type
        self.example2idx = example2idx

        self.evaluated = False

        self.ranks = None
        self.scores = None

    def evaluate(self, b_size, max_better=False):
        use_cuda = next(self.model.parameters()).is_cuda
        ent_idx = reindex_kg_idx_enteties_vector(self.kg.head_idx, self.kg, self.example2idx)
        concept_idx = reindex_kg_idx_enteties_vector(self.kg.tail_idx, self.kg, self.example2idx)
        r_idx = self.kg.relations

        unique_concepts_idx = torch.unique(concept_idx)
        n_con = self.model.n_concepts

        if use_cuda:
            unique_concepts_idx = unique_concepts_idx.cuda()

        ent_rel = torch.cat([ent_idx.view(-1, 1), r_idx.view(-1, 1)], dim=1).split(b_size)

        self.scores = torch.zeros(ent_idx.size(0), n_con)

        for i, batch in tqdm(enumerate(ent_rel), total=len(ent_rel),
                             unit='batch', desc='Cross evaluation'):

            ent_batch_idx = batch[:, 0]
            r_batch_idx = batch[:, 1]

            if use_cuda:
                ent_batch_idx = ent_batch_idx.cuda()
                r_batch_idx = r_batch_idx.cuda()
            # подготовим candidates к расчетам ранков
            e, c, r, candidates = self.model.inference_prepare_candidates(ent_batch_idx, unique_concepts_idx,
                                                                          r_batch_idx)
            # посчитаем scores между ent из батча и каждым concepts
            self.scores_batch = self.model.inference_scoring_function(e, candidates, r).cpu()
            del e
            del c
            del candidates
            del r
            gc.collect()
            torch.cuda.empty_cache()

            self.scores[i * b_size: (i + 1) * b_size, :] = self.scores_batch

        self.ranks = self.scores.argsort(dim=1, descending=max_better).cpu()
        # сохраним реиндексацию (это наш правильный ответ) для испольования в метриках
        self.concept_idx = concept_idx
        self.evaluated = True

    def hit_at_k(self, k=3):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'CrossEvaluator.evaluate')

        hit = (self.ranks[:, :k] == self.concept_idx.view(-1, 1)).sum(dim=1).float().mean()

        return hit.item()

    def mrr(self):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'CrossEvaluator.evaluate')

        mrr = torch.mean(1 / ((self.ranks == self.concept_idx.view(-1, 1)).int().argmax(dim=1) + 1))
        return mrr.item()


class JoieLinkPredictionEvaluator(LinkPredictionEvaluator):
    def __init__(self, joie_model, knowledge_graph, type_knowledge_graph):

        self.example2idx = joie_model.example2idx
        self.type_kg = type_knowledge_graph
        self.model = joie_model.models[type_knowledge_graph]
        self.kg = knowledge_graph

        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()

        self.evaluated = False

    def evaluate(self, b_size=128, candidates_b_size=100):
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()

        dict_of_tails = reindex_dict_of_enteties_kg(self.kg.dict_of_tails, self.kg, self.example2idx)
        dict_of_heads = reindex_dict_of_enteties_kg(self.kg.dict_of_heads, self.kg, self.example2idx)
        # реиндексация
        head_idx = reindex_kg_idx_enteties_vector(self.kg.head_idx, self.kg, self.example2idx)
        tail_idx = reindex_kg_idx_enteties_vector(self.kg.tail_idx, self.kg, self.example2idx)

        dataiter = torch.cat([
            head_idx.view(-1, 1),
            tail_idx.view(-1, 1),
            self.kg.relations.view(-1, 1)
        ], dim=1).split(b_size)

        for i, batch in tqdm(enumerate(dataiter), total=len(dataiter),
                             unit='batch',
                             desc=f'KG {self.type_kg}: Link prediction evaluation'):
            h_idx, t_idx, r_idx = batch[:, 0], batch[:, 1], batch[:, 2]

            if use_cuda:
                h_idx = h_idx.cuda()
                t_idx = t_idx.cuda()
                r_idx = r_idx.cuda()

            # здесь идет расчет по батчам оценки score

            h_emb, t_emb, r_emb, candidates = self.model.inference_prepare_candidates(h_idx, t_idx, r_idx,
                                                                                      entities=True)

            # true tails
            # разбиваем самую массивную часть на батчи
            cb_size = candidates_b_size
            candidates_batches = candidates.split(cb_size, dim=1)

            true_tails_scores = empty(size=(batch.size(0), self.model.n_ent))
            true_heads_scores = empty(size=(batch.size(0), self.model.n_ent))
            # начинаем итеративный расчет по частям кандидатов
            for j, candidates_batch in enumerate(candidates_batches):
                # true tails
                true_tails_batch_scores = self.model.inference_scoring_function(h_emb, candidates_batch, r_emb)
                # вставили в нужное место расчетные скоры
                true_tails_scores[:, j * cb_size: (j + 1) * cb_size] = true_tails_batch_scores
                # true heads
                true_heads_batch_scores = self.model.inference_scoring_function(candidates_batch, t_emb, r_emb)
                # вставили в нужное место расчетные скоры
                true_heads_scores[:, j * cb_size: (j + 1) * cb_size] = true_heads_batch_scores

                torch.cuda.empty_cache()
            # провели оценку scores
            if use_cuda:
                true_tails_scores = true_tails_scores.cuda()
                true_heads_scores = true_heads_scores.cuda()

            filt_scores = filter_scores(true_tails_scores,
                                        dict_of_tails,
                                        h_idx, r_idx, t_idx)
            self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(true_tails_scores, t_idx).detach()
            self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()

            # провели оценку scores
            filt_scores = filter_scores(true_heads_scores,
                                        dict_of_heads,
                                        t_idx, r_idx, h_idx)
            self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(true_heads_scores, h_idx).detach()
            self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()

            del h_emb
            del true_tails_scores
            del true_heads_scores
            del filt_scores
            del candidates
            del batch
            del t_emb
            del r_emb
            torch.cuda.empty_cache()

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()

    def hit_at_k_heads(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

        return head_hit.item(), filt_head_hit.item()

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

        return tail_hit.item(), filt_tail_hit.item()

    def hit_at_k(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        # head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return tail_hit, filt_tail_hit

    def mrr(self):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        # head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        tail_mrr = (self.rank_true_tails.float() ** (-1)).mean()
        # filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        filt_tail_mrr = (self.filt_rank_true_tails.float() ** (-1)).mean()

        return tail_mrr.item(), filt_tail_mrr.item()
