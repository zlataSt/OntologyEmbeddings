import torch
from torchkge.sampling import NegativeSampler
import random

class CrossNegativeSampler(NegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)

        self.unique_tails = kg.tail_idx.unique()

    def corrupt_batch(self, heads, tails, relations=None, n_neg=None):
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # здесь мы понимам, какое число надо выкинуть из рассмотрения
        neg_tails_help = neg_tails.view(-1, 1) - self.unique_tails

        # здесь в каждой строке будут варианты для случайного выбора неправильного концепта
        neg_tails_variants = neg_tails.view(-1, 1) - neg_tails_help[neg_tails_help != 0].view(
            neg_tails.size(0), self.unique_tails.size(0) - 1)

        neg_tails = neg_tails_variants[torch.arange(neg_tails_variants.size(0)),
        torch.randint(0, neg_tails_variants.size(1), size=(neg_tails_variants.size(0),))]

        return neg_heads.long(), neg_tails.long()

def is_consistent(triples, O):
    all_triples = set(triples)
    all_triples.update(O)
    entities = set()
    for t in all_triples:
        entities.add(t[0])
        entities.add(t[2])
    for e in entities:
        if (e, "owl:sameAs", e) not in all_triples:
            connected_entities = set([e])
            connected_triples = set()
            while connected_entities:
                cur_e = connected_entities.pop()
                for t in all_triples:
                    if t[0] == cur_e or t[2] == cur_e:
                        if t not in connected_triples:
                            connected_triples.add(t)
                            if t[0] not in connected_entities:
                                connected_entities.add(t[0])
                            if t[2] not in connected_entities:
                                connected_entities.add(t[2])
            for t1 in connected_triples:
                for t2 in connected_triples:
                    if t1 != t2 and t1[1] == t2[1] and t1[2] != t2[2]:
                        return False
    return True

def get_explanations(triples, O):
    explanations = set()
    for t in triples:
        for o in O:
            if t[1] == o[1] and t[2] != o[2]:
                explanations.add((t, o))
            elif t[2] == o[2] and t[1] != o[1]:
                explanations.add((t, o))
    return explanations

def GeneralizedSamples(beta, G, O, num_samples):
    # Получаем сущность o' из предсказанного триплета beta
    _, _, o_prime = beta

    # Получаем все предикаты из графа знаний G
    predicates_G = set([p for _, p, _ in G])

    # Получаем все предикаты из онтологии O
    predicates_O = set([p for _, p, _ in O])

    # Получаем все сущности, связанные с o' в графе знаний G
    entities_G = set([s for s, p, _ in G if p in predicates_G and o_prime == _])

    # Получаем все сущности, связанные с o' в онтологии O
    entities_O = set(s for s, p, _ in filter(lambda x: x[1] in predicates_O and x[2] == o_prime, O))

    # Получаем все сущности из корпуса
    entities_corpus = set([s for s, p, o in G.union(O)])

    # Получаем множество всех сущностей, не связанных с o' ни в графе знаний G, ни в онтологии O
    entities_not_related = entities_corpus - entities_G - entities_O - set([o_prime])

    # Сгенерируем num_samples отрицательных примеров
    negative_samples = set()
    for i in range(num_samples):
        # Выбираем случайный предикат, отличный от предикатов в G и O
        predicate = None
        while predicate is None or predicate in predicates_G or predicate in predicates_O:
            predicate = random.choice(list(predicates_G.union(predicates_O).union(set([beta[1]]))))

        # Выбираем случайную сущность, которая не связана с o' в G и O
        predicates_O.difference_update(self, s="")