import torch
from torchkge.utils import MarginLoss

# Define the TransO model class
class TransO(torch.nn.Module):
    def __init__(self, kg, embedding_dim):
        super().__init__()

        # Embedding dimensionality
        self.embedding_dim = embedding_dim

        # Entity and relation embeddings
        self.entity_embeddings = torch.nn.Embedding(kg.num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(kg.num_relations, embedding_dim)

        # Projection matrices for entity types and relation types
        self.entity_type_projections = torch.nn.Embedding(kg.num_entity_types, embedding_dim * embedding_dim)
        self.relation_type_projections = torch.nn.Embedding(kg.num_relation_types, embedding_dim * embedding_dim)

        # Initialize all embeddings and projections randomly
        self.entity_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.relation_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.entity_type_projections.weight.data.uniform_(-0.1, 0.1)
        self.relation_type_projections.weight.data.uniform_(-0.1, 0.1)

        # Loss function
        self.loss_fn = MarginLoss(margin=1.0)

    def forward(self, heads, relations, tails, entity_types, relation_types):
        # Look up embeddings for heads, relations, and tails
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # Look up projection matrices for entity types and relation types
        entity_type_projections = self.entity_type_projections(entity_types)
        relation_type_projections = self.relation_type_projections(relation_types)

        # Reshape projection matrices to be 3D tensors
        entity_type_projections = entity_type_projections.view(-1, self.embedding_dim, self.embedding_dim)
        relation_type_projections = relation_type_projections.view(-1, self.embedding_dim, self.embedding_dim)

        # Project entity embeddings with entity type projections
        projected_head_embeddings = torch.bmm(entity_type_projections, head_embeddings.unsqueeze(-1)).squeeze()
        projected_tail_embeddings = torch.bmm(entity_type_projections, tail_embeddings.unsqueeze(-1)).squeeze()

        # Project relation embeddings with relation type projections
        projected_relation_embeddings = torch.bmm(relation_type_projections, relation_embeddings.unsqueeze(-1)).squeeze()

        # Compute distance scores with projected embeddings and relation embeddings
        scores = torch.norm(projected_head_embeddings + projected_relation_embeddings - projected_tail_embeddings, p=2, dim=1)

        return scores

    def compute_loss(self, pos_triples, neg_triples):
        # Compute scores for positive and negative triples
        pos_scores = self(*pos_triples)
        neg_scores = self(*neg_triples)

        # Compute loss
        loss = self.loss_fn(pos_scores, neg_scores)

        return loss

# Define the training function
def train_transo(kg, embedding_dim, num_epochs, batch_size, learning_rate):
    # Create the TransO model
    model = TransO(kg, embedding_dim)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Number of batches
    num_batches = kg.num_triples // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_num in range(num_batches):
            # Sample positive and negative triples for the current batch
            pos_triples = kg.sample(batch_size)
            neg_triples = kg.sample(batch_size, negative=True)

            # Compute loss
            loss = model.compute_loss(pos_triples, neg_triples)
            epoch_loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        print("Epoch %d / %d : Average loss = %f" % (epoch+1, num_epochs, epoch_loss / num_batches))

        # Update the entity embeddings with type projections
        with torch.no_grad():
            for entity_type_id in range(kg.num_entity_types):
                # Select all entity embeddings for the current entity type
                entity_ids = torch.nonzero(kg.entity_type == entity_type_id, as_tuple=True)[0]
                entity_embeddings = model.entity_embeddings(torch.LongTensor(entity_ids)).squeeze()

                # Select the projection matrix for the current entity type
                entity_type_projection = model.entity_type_projections(torch.LongTensor([entity_type_id])).squeeze().view(embedding_dim, embedding_dim)

                # Project the entity embeddings with the entity type projection
                projected_entity_embeddings = torch.mm(entity_embeddings, entity_type_projection)

                # Update the entity embeddings with the projected entity embeddings
                model.entity_embeddings.weight.data[entity_ids] = projected_entity_embeddings.cpu()

        # Update the relation embeddings with type projections
        with torch.no_grad():
            for relation_type_id in range(kg.num_relation_types):
                # Select all relation embeddings for the current relation type
                relation_ids = torch.nonzero(kg.relation_type == relation_type_id, as_tuple=True)[0]
                relation_embeddings = model.relation_embeddings(torch.LongTensor(relation_ids)).squeeze()

                # Select the projection matrix for the current relation type
                relation_type_projection = model.relation_type_projections(torch.LongTensor([relation_type_id])).squeeze().view(embedding_dim, embedding_dim)

                # Project the relation embeddings with the relation type projection
                projected_relation_embeddings = torch.mm(relation_embeddings, relation_type_projection)

                # Update the relation embeddings with the projected relation embeddings
                model.relation_embeddings.weight.data[relation_ids] = projected_relation_embeddings.cpu()
    return model

