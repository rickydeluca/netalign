import torch
import torch.nn as nn
import torch.optim as optim

from netalign.data.utils import dict_to_perm_mat
from netalign.models.shelley.embedding import init_embedding_module
from netalign.models.shelley.feat_init import init_feat_module
from netalign.models.shelley.matching import init_matching_module

'''
class SHELLEY_G(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the Frankenstein's Monster.
        """
        """
        Initialize the Frankenstein's Monster.
        """
        super(SHELLEY, self).__init__()
        
        # Init modules
        self.f_init = init_feat_module(cfg.FEATS)
        self.f_update = init_embedding_module(cfg.EMBEDDING)
        self.model = init_matching_module(self.f_update, cfg.MATCHING)

        # Configure training
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE

        if cfg.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.L2NORM
            )
        elif cfg.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM)
        else:
            raise ValueError(f"Invalid optimizer: {cfg.TRAIN.OPTIMIZER}")

    def align(self, input_data, train=False, grad=True):
        """
        Predict alignment between a graph pair.
        """
        # Read input
        device = next(iter(self.model.parameters())).device
        input_data = move_tensors_to_device(input_data, device=device)
        graph_s = input_data['graph_pair'][0]
        graph_t = input_data['graph_pair'][1]
        gt_perm = input_data['gt_perm']
        src_ns = torch.tensor([graph_s.num_nodes])
        tgt_ns = torch.tensor([graph_t.num_nodes])

        # Init features
        graph_s.x = self.f_init(graph_s).to(device)
        graph_t.x = self.f_init(graph_t).to(device)

        # Predict alignment
        if train:
            if isinstance(self.model, BatchStableGM):   # Batching on graph pair
                # Prepare for mini-batching
                num_batches = (graph_s.num_nodes + self.batch_size - 1) // self.batch_size
                _, src_indices, tgt_indices = gt_perm.nonzero(as_tuple=True)
                
                if grad:
                    shuffling = torch.randperm(len(src_indices))
                else:
                    shuffling = torch.arange(len(src_indices))    # Do not shuffle during validation
                shuff_src_indices = src_indices[shuffling]
                shuff_tgt_indices = tgt_indices[shuffling]
                
                # Mini-batching loop
                loss = 0.0
                for b_iter in range(num_batches):
                    src_train_batch = shuff_src_indices[b_iter * self.batch_size:(b_iter + 1) * self.batch_size]
                    tgt_train_batch = shuff_tgt_indices[b_iter * self.batch_size:(b_iter + 1) * self.batch_size]

                    _, batch_loss = self.model(
                        graph_s,
                        graph_t,
                        train=True,
                        batch_indices_s=src_train_batch,
                        batch_indices_t=tgt_train_batch
                    )

                    if grad:
                        self.optimizer.zero_grad()
                        batch_loss.backward()
                        self.optimizer.step()

                    loss += batch_loss / num_batches

            else:   # No batching on graph pair     
                _, loss = self.model(graph_s, graph_t, src_ns=src_ns, tgt_ns=tgt_ns, train=True, gt_perm=gt_perm)
            
            return loss
        
        # Evaluation
        else:
            S = self.model(graph_s, graph_t, train=False)
            return S
        
    def train_eval(self, train_loader, val_loader):
        """
        Train and validate model.
        """
        best_val_loss = float('inf')
        best_state_dict = {}
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train(train_loader)

            # Eval
            val_loss = self.evaluate(val_loader, use_acc=False)

            print(f"Epoch: {epoch+1}/{self.epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            # Update best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.model.state_dict()
                best_epoch = epoch+1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check for early stop
            if patience_counter > self.patience:
                print(f"Early stop triggered after {epoch+1} epochs!")
                break
                

        # Load best state dict
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

        return best_val_loss, best_epoch

    def train(self, train_loader):
        """
        Train model.
        """
        self.model.train()
        n_batches = len(train_loader)
        train_loss = 0
        for pair_dict in train_loader:
            
            # Foraward step
            loss = self.align(pair_dict, train=True)

            if not isinstance(self.model, BatchStableGM):   # Backward already performed if using `BatchStableGM`
                # Backward step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss += loss / n_batches
        
        return train_loss
    
    @torch.no_grad()
    def evaluate(self, eval_loader, use_acc=False):
        """
        Evaluate model (validation/test).
        """
        self.model.eval()
        n_batches = len(eval_loader)
        if use_acc:
            val_acc = 0
            for pair_dict in eval_loader:
                S = self.align(pair_dict, train=False)
                batch_size = S.shape[0]
                for i in range(batch_size):
                    pred_mat = greedy_match(S[i].detach().cpu().numpy())
                    eval_gt = pair_dict['gt_perm'][i].detach().cpu().numpy()
                    val_acc += compute_accuracy(pred_mat, eval_gt) / batch_size / n_batches
        
            return val_acc 

        else:
            val_loss = 0
            for pair_dict in eval_loader:
                loss = self.align(pair_dict, train=True, grad=False)
                val_loss += loss.item() / n_batches
            
            return val_loss
'''

class SHELLEY(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the Frankenstein's Monster.
        """
        super(SHELLEY, self).__init__()
        
        # Init modules
        self.f_init = init_feat_module(cfg.FEATS)
        self.f_update = init_embedding_module(cfg.EMBEDDING)
        self.model = init_matching_module(self.f_update, cfg.MATCHING)

        # Configure training
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE

        if cfg.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.L2NORM
            )
        elif cfg.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM)
        else:
            raise ValueError(f"Invalid optimizer: {cfg.TRAIN.OPTIMIZER}")
        
    def align(self, pair_dict):
        # Read input
        self.graph_s = pair_dict['graph_pair'][0]
        self.graph_t = pair_dict['graph_pair'][1]
        self.device = self.graph_s.edge_index.device
        self.src_ns = torch.tensor([self.graph_s.num_nodes]).to(self.device)
        self.tgt_ns = torch.tensor([self.graph_t.num_nodes]).to(self.device)
        self.gt_train = dict_to_perm_mat(pair_dict['gt_train'], self.graph_s.num_nodes, self.graph_t.num_nodes).to(self.device).unsqueeze(0)
        self.gt_val = dict_to_perm_mat(pair_dict['gt_val'], self.graph_s.num_nodes, self.graph_t.num_nodes).to(self.device).unsqueeze(0)

        # Init features
        self.init_features()
        
        # Train model
        self.best_val_epoch = self.train_eval()

        # Get alignment
        self.S = self.get_alignment()

        return self.S, self.best_val_epoch

    def init_features(self):
        device = self.graph_s.edge_index.device
        self.graph_s.x = self.f_init(self.graph_s).to(device)
        self.graph_t.x = self.f_init(self.graph_t).to(device)
        
    def train_eval(self):
        best_val_loss = float('inf')
        best_val_epoch = 0
        best_state_dict = {}
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train()

            # Eval
            val_loss = self.evaluate()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check for early stop
            if patience_counter > self.patience:
                print(f"Early stop triggered after {epoch+1} epochs!")
                break

            print(f"Epoch: {epoch+1}, Train loss: {train_loss}, Val loss: {val_loss}")

        # Load best model
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

        return best_val_epoch


    def train(self):
        self.model.train()

        # Forward
        train_dict = {'gt_perm': self.gt_train, 'src_ns': self.src_ns, 'tgt_ns': self.tgt_ns}
        _, train_loss = self.model(self.graph_s, self.graph_t, train=True, train_dict=train_dict)

        # Backward
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return train_loss
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        train_dict = {'gt_perm': self.gt_val, 'src_ns': self.src_ns, 'tgt_ns': self.tgt_ns}
        _, val_loss = self.model(self.graph_s, self.graph_t, train=True, train_dict=train_dict)
        return val_loss
    
    @torch.no_grad()
    def get_alignment(self):
        self.model.eval()
        S = self.model(self.graph_s, self.graph_t).squeeze(0).detach().cpu().numpy()
        return S
