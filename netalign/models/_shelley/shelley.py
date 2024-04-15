import torch
import torch.nn as nn
import torch.optim as optim

from netalign.data.utils import move_tensors_to_device
from netalign.evaluation.matcher import greedy_match
from netalign.evaluation.metrics import compute_accuracy
from netalign.models.shelley.embedding import GCN, GINE, GINE2
from netalign.models.shelley.feat_init import Degree, Share
from netalign.models.shelley.matching import StableGM


class SHELLEY_G(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the Frankenstein's Monster.
        """
        super(SHELLEY_G, self).__init__()

        # MODEL PARAMETERS

        # Init features
        if cfg.INIT.FEATURES.lower() == 'degree':
            self.f_init = Degree()
        elif cfg.INIT.FEATURES.lower() == 'share':
            self.f_init = Share(cfg.INIT.FEATURE_DIM)
        else:
            raise ValueError(f"Invalid features: {cfg.MODEL.FEATURES}.")

        # Embedding model
        if cfg.EMBEDDING.MODEL == 'gcn':
            f_update = GCN(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                hidden_channels=cfg.EMBEDDING.HIDDEN_CHANNELS,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_layers=cfg.EMBEDDING.NUM_LAYERS
            )
        elif cfg.EMBEDDING.MODEL == 'gine':
            f_update = GINE(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                dim=cfg.EMBEDDING.DIM,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_conv_layers=cfg.EMBEDDING.NUM_CONV_LAYERS
            )
        elif cfg.EMBEDDING.MODEL == 'gine2':
            f_update = GINE2(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                dim=cfg.EMBEDDING.DIM,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_conv_layers=cfg.EMBEDDING.NUM_CONV_LAYERS
            )
        else:
            raise ValueError(f"Invalid embedding model: {cfg.EMBEDDING.MODEL}")
        
        # Matching model
        if cfg.MATCHING.MODEL == 'sgm':
            self.model = StableGM(
                f_update=f_update,
                beta=cfg.MATCHING.BETA,
                n_sink_iters=cfg.MATCHING.N_SINK_ITERS,
                tau=cfg.MATCHING.TAU
            )
        else:
            raise ValueError(f"Invalid matching model: {cfg.MATCHING.MODEL}")
        
        # TRAINING PARAMETERS

        # Optimizer
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
        
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
    
    def align(self, input_data, train=False):
        """
        Predict alignment between a graph pair.
        """
        # Read input
        device = next(iter(self.model.parameters())).device
        input_data = move_tensors_to_device(input_data, device=device)
        graph_s = input_data['graph_pair'][0]
        graph_t = input_data['graph_pair'][1]
        gt_matrix = input_data['gt_matrix']

        # Init features
        graph_s.x = self.f_init(graph_s).to(device)
        graph_t.x = self.f_init(graph_t).to(device)
        
        # Predict alignment
        if train:
            S, loss = self.model(graph_s, graph_t, train=True, gt_perm=gt_matrix)
            return S, loss
        else:
            S = self.model(graph_s, graph_t, train=False)
            return S
        
    def train_eval(self, train_loader, val_loader):
        """
        Train and validate model.
        """
        best_val_acc = 0
        min_thresh = 0.1
        best_state_dict = {}
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train(train_loader)

            # Eval
            val_acc = self.evaluate(val_loader, use_acc=True)

            print(f"Epoch: {epoch+1}/{self.epochs}, Train loss: {train_loss:.4f}, Val accuracy: {val_acc:.4f}")

            # Update best model
            if val_acc > best_val_acc and val_acc > min_thresh:
                best_val_acc = val_acc
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

        return best_val_acc, best_epoch

    def train(self, train_loader):
        """
        Train model.
        """
        self.model.train()
        n_batches = len(train_loader)
        train_loss = 0
        for pair_dict in train_loader:
            # Foraward step
            _, loss = self.align(pair_dict, train=True)

            # Backward step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss
        
        return train_loss / n_batches
    
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
                S = self.align(pair_dict, train=False).squeeze(0).detach().cpu().numpy()
                pred_mat = greedy_match(S)
                eval_gt = pair_dict['gt_matrix'].squeeze(0).detach().cpu().numpy()
                acc = compute_accuracy(pred_mat, eval_gt)
                val_acc += acc
            return val_acc / n_batches

        else:
            val_loss = 0
            for pair_dict in eval_loader:
                _, loss = self.align(pair_dict, train=True)
                val_loss += loss.item()
            
            return val_loss / n_batches


class SHELLEY_N(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the Frankenstein's Monster.
        """
        super(SHELLEY_N, self).__init__()
        
        # Model parameters:

        # Init features
        if cfg.INIT.FEATURES.lower() == 'degree':
            self.f_init = Degree()
        elif cfg.INIT.FEATURES.lower() == 'share':
            self.f_init = Share(cfg.INIT.FEATURE_DIM)
        else:
            raise ValueError(f"Invalid features: {cfg.MODEL.FEATURES}.")

        # Embedding model
        if cfg.EMBEDDING.MODEL == 'gcn':
            f_update = GCN(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                hidden_channels=cfg.EMBEDDING.HIDDEN_CHANNELS,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_layers=cfg.EMBEDDING.NUM_LAYERS
            )
        elif cfg.EMBEDDING.MODEL == 'gine':
            f_update = GINE(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                dim=cfg.EMBEDDING.DIM,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_conv_layers=cfg.EMBEDDING.NUM_CONV_LAYERS
            )
        elif cfg.EMBEDDING.MODEL == 'gine2':
            f_update = GINE2(
                in_channels=cfg.EMBEDDING.IN_CHANNELS,
                dim=cfg.EMBEDDING.DIM,
                out_channels=cfg.EMBEDDING.OUT_CHANNELS,
                num_conv_layers=cfg.EMBEDDING.NUM_CONV_LAYERS
            )
        else:
            raise ValueError(f"Invalid embedding model: {cfg.EMBEDDING.MODEL}")
        
        # Matching model
        if cfg.MATCHING.MODEL == 'sgm':
            self.model = StableGM(
                f_update=f_update,
                beta=cfg.MATCHING.BETA,
                n_sink_iters=cfg.MATCHING.N_SINK_ITERS,
                tau=cfg.MATCHING.TAU
            )
        else:
            raise ValueError(f"Invalid matching model: {cfg.MATCHING.MODEL}")
        
        # Traning parameters:
        
        # Optimizer
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
        
        # Epochs
        self.epochs = cfg.TRAIN.EPOCHS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.patience = cfg.TRAIN.PATIENCE

    def align(self, input_data):
        # Read input
        self.graph_s = input_data['graph_pair'][0]
        self.graph_t = input_data['graph_pair'][1]
        self.train_dict = input_data['train_dict']
        self.val_dict = input_data['val_dict']

        self.init_features()

        # Load graphs to the model
        self.model.graph_s = self.graph_s
        self.model.graph_t = self.graph_t
        
        # Train model
        self.best_val_epoch = self.train_eval()

        # Get alignment
        self.S = self.get_alignment()

        return self.S, self.best_val_epoch

    def init_features(self):
        self.device = self.graph_s.edge_index.device
        self.graph_s.x = self.f_init(self.graph_s).to(self.device)
        self.graph_t.x = self.f_init(self.graph_t).to(self.device)
        
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

        # Mini-batching
        shuff = torch.randperm(len(self.train_dict))
        source_indices = torch.LongTensor(list(self.train_dict.keys()))[shuff].to(self.device)
        target_indices = torch.LongTensor(list(self.train_dict.values()))[shuff].to(self.device)
        
        num_iters = len(source_indices) // self.batch_size
        assert num_iters > 0, "Too large batch_size!"
        if len(source_indices) % self.batch_size > 0:
            num_iters += 1

        train_loss = 0
        for iter in range(num_iters):
            batch_s = source_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
            batch_t = target_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
            
            loss = self.model(batch_s, batch_t, train=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss / num_iters

        return train_loss
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        # Mini-batching
        shuff = torch.randperm(len(self.val_dict))
        source_indices = torch.LongTensor(list(self.val_dict.keys()))[shuff].to(self.device)
        target_indices = torch.LongTensor(list(self.val_dict.values()))[shuff].to(self.device)
        
        num_iters = len(source_indices) // self.batch_size
        assert num_iters > 0, "Too large batch_size!"
        if len(source_indices) % self.batch_size > 0:
            num_iters += 1

        val_loss = 0
        for iter in range(num_iters):
            batch_s = source_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
            batch_t = target_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
            
            loss = self.model(batch_s, batch_t, train=True)
            val_loss += loss / num_iters

        return val_loss
    
    @torch.no_grad()
    def get_alignment(self):
        self.model.eval()
        S = self.model.get_alignment().squeeze(0).detach().cpu().numpy()
        return S
