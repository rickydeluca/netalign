import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


class PaleEmbedding(nn.Module):
    def __init__(self, n_nodes, embedding_dim, deg, neg_sample_size): #, cuda):
        super(PaleEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.deg = deg
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = EmbeddingLossFunctions()
        self.n_nodes = n_nodes
        # self.use_cuda = cuda


    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss, loss0, loss1 = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        loss0 = loss0/batch_size
        loss1 = loss1/batch_size
        
        return loss, loss0, loss1


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            # if self.use_cuda:
            #     neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            neighbor_output = F.normalize(neighbor_output, dim=1)
            neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output

    def get_embedding(self):
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        # if self.use_cuda:
        #     nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512
        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding
    

class EmbeddingLossFunctions(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be
                based on dot product.
        """
        self.neg_sample_weights = neg_sample_weights
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")


    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [n_batch_edges x feature_size].
        """
        # shape: [n_batch_edges, input_dim1]
        result = torch.sum(inputs1 * inputs2, dim=1) # shape: (n_batch_edges,)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [n_batch_edges x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = inputs1.mm(neg_samples.t()) #(n_batch_edges, num_neg_samples)
        return neg_aff


    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        inputs1: Tensor (512, 256), normalized vector
        inputs2: Tensor (512, 256), normalized vector
        neg_sample: Tensor (20, 256)
        """
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_labels = torch.ones(true_aff.shape)  # (n_batch_edges,)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss0 = true_xent.sum()
        loss1 = self.neg_sample_weights * neg_xent.sum()
        loss = loss0 + loss1
        return loss, loss0, loss1