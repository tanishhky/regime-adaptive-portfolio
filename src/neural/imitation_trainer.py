import torch
import torch.nn.functional as F
import numpy as np

class ImitationTrainer:
    """
    Trains the policy network to mimic the hindsight oracle via supervised learning.

    Loss = MSE(policy_weights, oracle_optimal_weights) + entropy_bonus

    This runs BEFORE PPO fine-tuning. The oracle provides a warm-start;
    PPO then adapts online from actual trading experience.

    Training schedule at each walk-forward rebalance:
        1. Generate oracle training pairs from the training window
        2. Train policy via MSE for N_imitation_epochs
        3. Then switch to PPO for online adaptation during the test window
    """

    def __init__(self, policy: torch.nn.Module, lr: float = 1e-3, n_epochs: int = 30):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
        self.n_epochs = n_epochs

    def train(self, oracle_pairs: list[tuple[np.ndarray, np.ndarray]],
              val_split: float = 0.2) -> dict[str, list[float]]:
        """
        Train policy to imitate the oracle.

        Args:
            oracle_pairs: list of (state, oracle_weights) tuples
            val_split: fraction held out for validation (early stopping, patience=5)

        Returns:
            dict with 'train_loss' and 'val_loss' histories
        """
        if not oracle_pairs:
            return {'train_loss': [], 'val_loss': []}

        # Shuffle and split
        np.random.shuffle(oracle_pairs)
        split_idx = int(len(oracle_pairs) * (1 - val_split))
        train_pairs = oracle_pairs[:split_idx]
        val_pairs = oracle_pairs[split_idx:]

        def _make_tensors(pairs):
            states = torch.FloatTensor(np.array([p[0] for p in pairs]))
            targets = torch.FloatTensor(np.array([p[1] for p in pairs]))
            return states, targets

        train_states, train_targets = _make_tensors(train_pairs)
        val_states, val_targets = _make_tensors(val_pairs)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        self.policy.train()
        for epoch in range(self.n_epochs):
            # Train step
            self.optimizer.zero_grad()
            self.policy.reset_hidden(train_states.size(0))
            
            # Policy forward expects shape (batch, seq_len, state_dim)
            # Add seq_len dimension
            inputs = train_states.unsqueeze(1)
            
            # forward might return diff tuple lengths depending on single or multi-head
            out = self.policy(inputs)
            policy_weights = out[0].squeeze(1) # shape (batch, n_assets)
            
            # The target might contain cash weight at the end.
            # Our policy networks output n_assets (11).
            # If target has n_assets+1, we truncate the cash weight for MSE.
            # Cash is managed by CashManager manually!
            n_assets = policy_weights.size(1)
            target_w = train_targets[:, :n_assets]
            
            # Normalize target_w so it sums to 1
            target_w = target_w / (target_w.sum(dim=-1, keepdim=True) + 1e-8)

            loss = F.mse_loss(policy_weights, target_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            train_loss = loss.item()

            # Val step
            self.policy.eval()
            with torch.no_grad():
                if len(val_states) > 0:
                    self.policy.reset_hidden(val_states.size(0))
                    val_inputs = val_states.unsqueeze(1)
                    val_out = self.policy(val_inputs)
                    val_policy_weights = val_out[0].squeeze(1)
                    
                    val_target_w = val_targets[:, :n_assets]
                    val_target_w = val_target_w / (val_target_w.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    v_loss = F.mse_loss(val_policy_weights, val_target_w).item()
                else:
                    v_loss = train_loss
                    
            history['train_loss'].append(train_loss)
            history['val_loss'].append(v_loss)

            self.policy.train()

            # Early stopping
            if v_loss < best_val_loss - 1e-5:
                best_val_loss = v_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return history
