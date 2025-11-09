# Practical Examples: Distributed Systems for AI

This document contains runnable code examples that demonstrate distributed systems concepts applied to AI training.

## Example 1: Simple Data Parallelism (MapReduce-style)

### Concept
Train a model by distributing data batches across multiple workers, just like MapReduce distributes map tasks.

### Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from typing import List
import time

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DataParallelTrainer:
    """
    MapReduce-inspired data parallel training
    - Map: Each worker computes gradients
    - Reduce: Aggregate gradients
    """
    def __init__(self, model, num_workers=4):
        self.model = model
        self.num_workers = num_workers
        self.workers = []

    def map_gradients(self, worker_id, data_batch, model_params):
        """
        MAP PHASE: Compute gradients on a data batch
        Like: map(doc) → emit(word, 1)
        """
        # Clone model with shared parameters
        model = SimpleModel()
        model.load_state_dict(model_params)

        # Forward pass
        inputs, targets = data_batch
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # Backward pass
        loss.backward()

        # Return gradients (like Map emits key-value pairs)
        return {name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.grad is not None}

    def reduce_gradients(self, all_gradients: List[dict]) -> dict:
        """
        REDUCE PHASE: Aggregate gradients from all workers
        Like: reduce(word, [1,1,1]) → (word, 3)
        """
        if not all_gradients:
            return {}

        # Average gradients
        reduced = {}
        for name in all_gradients[0].keys():
            grads = [worker_grads[name] for worker_grads in all_gradients]
            reduced[name] = torch.stack(grads).mean(dim=0)

        return reduced

    def train_step(self, data_loader):
        """
        Complete MapReduce round:
        1. Distribute data to workers (like input splits)
        2. Map: Compute gradients in parallel
        3. Shuffle & Sort: Collect gradients
        4. Reduce: Aggregate gradients
        5. Apply to model
        """
        # Get batches for all workers
        batches = []
        for i, batch in enumerate(data_loader):
            batches.append(batch)
            if len(batches) == self.num_workers:
                break

        # MAP PHASE: Parallel gradient computation
        model_params = self.model.state_dict()
        all_gradients = []

        # Simulate parallel execution
        with mp.Pool(self.num_workers) as pool:
            results = []
            for worker_id, batch in enumerate(batches):
                result = pool.apply_async(
                    self.map_gradients,
                    (worker_id, batch, model_params)
                )
                results.append(result)

            # Collect results (like MapReduce shuffle)
            all_gradients = [r.get() for r in results]

        # REDUCE PHASE: Aggregate gradients
        averaged_gradients = self.reduce_gradients(all_gradients)

        # Apply gradients to model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in averaged_gradients:
                    param.grad = averaged_gradients[name]

        return len(batches)

# Usage example
if __name__ == "__main__":
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __len__(self): return 1000
        def __getitem__(self, idx):
            return torch.randn(784), torch.randint(0, 10, (1,)).item()

    model = SimpleModel()
    trainer = DataParallelTrainer(model, num_workers=4)
    loader = DataLoader(DummyDataset(), batch_size=32)

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(3):
        batches_processed = trainer.train_step(loader)
        optimizer.step()
        print(f"Epoch {epoch}: Processed {batches_processed} batches")
```

### Key Takeaways
- **Map** = Forward + backward pass on each worker
- **Reduce** = Gradient aggregation
- **Coordinator** = Main process that distributes work
- **Fault tolerance** = Can add worker failure detection (like Lab 1)

---

## Example 2: Parameter Server (Lab 2 KV-style)

### Concept
Centralized parameter storage where workers pull parameters, compute gradients, and push updates.

### Implementation

```python
import torch
import threading
from queue import Queue
from typing import Dict

class ParameterServer:
    """
    Like Lab 2 KVServer:
    - Get(key) → Pull(layer_name)
    - Put(key, value) → Push(layer_name, gradient)
    """
    def __init__(self, model):
        self.params = {name: param.data.clone()
                      for name, param in model.named_parameters()}
        self.versions = {name: 0 for name in self.params.keys()}
        self.lock = threading.Lock()  # Like locks in kvsrv1/lock

    def pull(self, param_name: str) -> torch.Tensor:
        """Like Get() in Lab 2"""
        with self.lock:
            return self.params[param_name].clone()

    def push(self, param_name: str, gradient: torch.Tensor, lr: float = 0.01):
        """Like Put() in Lab 2"""
        with self.lock:
            # Apply gradient
            self.params[param_name] -= lr * gradient
            self.versions[param_name] += 1

    def pull_with_version(self, param_name: str) -> tuple:
        """Like linearizable read"""
        with self.lock:
            return self.params[param_name].clone(), self.versions[param_name]

    def push_if_version(self, param_name: str, gradient: torch.Tensor,
                       expected_version: int, lr: float = 0.01) -> bool:
        """
        Conditional update: only apply if version matches
        Prevents stale gradient application
        """
        with self.lock:
            if self.versions[param_name] == expected_version:
                self.params[param_name] -= lr * gradient
                self.versions[param_name] += 1
                return True
            return False

class PSWorker:
    """Worker that communicates with parameter server"""
    def __init__(self, worker_id: int, ps: ParameterServer, model):
        self.worker_id = worker_id
        self.ps = ps
        self.model = model

    def train_step(self, batch, sync=True):
        """
        Train step with parameter server

        sync=True: Synchronous SGD (linearizable)
        sync=False: Asynchronous SGD (eventual consistency)
        """
        inputs, targets = batch

        # 1. PULL: Get latest parameters
        for name, param in self.model.named_parameters():
            if sync:
                # Synchronous: get version too
                param.data, version = self.ps.pull_with_version(name)
            else:
                # Asynchronous: just pull (might be stale!)
                param.data = self.ps.pull(name)

        # 2. COMPUTE: Forward + backward
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()

        # 3. PUSH: Send gradients to server
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if sync:
                    # Synchronous: conditional update
                    success = self.ps.push_if_version(
                        name, param.grad, version
                    )
                    if not success:
                        print(f"Worker {self.worker_id}: Stale gradient rejected")
                else:
                    # Asynchronous: always apply
                    self.ps.push(name, param.grad)

        return loss.item()

# Usage
def run_ps_training(num_workers=4, sync=True):
    """Run parameter server training"""
    # Create parameter server
    master_model = SimpleModel()
    ps = ParameterServer(master_model)

    # Create workers
    workers = [PSWorker(i, ps, SimpleModel())
               for i in range(num_workers)]

    # Simulate training
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=32)

    for epoch in range(3):
        for batch in loader:
            # Each worker trains (in parallel in real implementation)
            for worker in workers:
                loss = worker.train_step(batch, sync=sync)

        print(f"Epoch {epoch} complete")
        print(f"Param versions: {ps.versions}")

if __name__ == "__main__":
    print("=== Synchronous Training (Linearizable) ===")
    run_ps_training(num_workers=4, sync=True)

    print("\n=== Asynchronous Training (Eventual Consistency) ===")
    run_ps_training(num_workers=4, sync=False)
```

### Key Takeaways
- **Pull/Push** = Get/Put from Lab 2
- **Versions** = Detect stale gradients
- **Locks** = Prevent race conditions (like kvsrv1/lock)
- **Sync vs Async** = Strong vs eventual consistency

---

## Example 3: Federated Learning (Raft-inspired)

### Concept
Multiple clients train locally, coordinator aggregates updates (like Raft leader coordinates log replication).

### Implementation

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FederatedCoordinator:
    """
    Like Raft leader:
    - Coordinates rounds (like terms)
    - Collects updates (like log entries)
    - Commits when quorum reached (like Raft commit)
    """
    def __init__(self, model, min_clients=3):
        self.global_model = model
        self.round = 0  # Like Raft term
        self.min_clients = min_clients  # Like quorum size
        self.client_updates = {}

    def start_round(self) -> Dict:
        """
        Start new round (like new Raft term)
        Send current model to clients (like log replication)
        """
        self.round += 1
        self.client_updates = {}

        return {
            'round': self.round,
            'model_state': copy.deepcopy(self.global_model.state_dict())
        }

    def receive_update(self, client_id: int, update: Dict) -> bool:
        """
        Receive update from client (like AppendEntries response)
        """
        if update['round'] != self.round:
            print(f"Rejecting stale update from client {client_id}")
            return False

        self.client_updates[client_id] = update
        return True

    def aggregate_and_commit(self) -> bool:
        """
        Aggregate updates if quorum reached (like Raft commit)

        Returns: True if committed, False if waiting for more clients
        """
        if len(self.client_updates) < self.min_clients:
            print(f"Only {len(self.client_updates)}/{self.min_clients} clients, waiting...")
            return False

        # Quorum reached! Aggregate updates
        print(f"Quorum reached ({len(self.client_updates)} clients), aggregating...")

        # Average model updates (FedAvg algorithm)
        aggregated_state = {}
        for param_name in self.global_model.state_dict().keys():
            param_updates = [
                update['model_state'][param_name]
                for update in self.client_updates.values()
            ]
            aggregated_state[param_name] = torch.stack(param_updates).mean(dim=0)

        # Commit: update global model
        self.global_model.load_state_dict(aggregated_state)

        print(f"Round {self.round} committed!")
        return True

class FederatedClient:
    """
    Like Raft peer:
    - Receives model from leader
    - Trains locally (like applying command to state machine)
    - Sends update back to leader
    """
    def __init__(self, client_id: int, local_data):
        self.client_id = client_id
        self.local_data = local_data
        self.local_model = None

    def train_local(self, round_info: Dict, epochs=1) -> Dict:
        """
        Train on local data (like Raft applying to state machine)
        """
        # Initialize model from coordinator (like Raft log replication)
        self.local_model = SimpleModel()
        self.local_model.load_state_dict(round_info['model_state'])

        # Train on local data
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)
        for epoch in range(epochs):
            for batch in self.local_data:
                inputs, targets = batch
                outputs = self.local_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Return update (like Raft response)
        return {
            'client_id': self.client_id,
            'round': round_info['round'],
            'model_state': self.local_model.state_dict()
        }

def simulate_federated_learning():
    """Simulate federated learning across multiple clients"""
    # Create global model
    global_model = SimpleModel()

    # Create coordinator (like Raft leader)
    coordinator = FederatedCoordinator(global_model, min_clients=3)

    # Create clients with local data (like Raft peers)
    clients = []
    for i in range(5):
        # Each client has different local data (privacy preserved!)
        local_data = DataLoader(DummyDataset(), batch_size=32)
        clients.append(FederatedClient(i, local_data))

    # Run federated learning rounds
    for round_num in range(3):
        print(f"\n{'='*50}")
        print(f"Starting Round {round_num + 1}")
        print(f"{'='*50}")

        # 1. Coordinator starts round (like Raft leader)
        round_info = coordinator.start_round()

        # 2. Clients train locally (like Raft peers)
        for client in clients[:4]:  # Simulate only 4 out of 5 responding
            update = client.train_local(round_info, epochs=1)

            # 3. Send update to coordinator
            coordinator.receive_update(client.client_id, update)

        # 4. Coordinator aggregates and commits (like Raft commit)
        committed = coordinator.aggregate_and_commit()

        if committed:
            print(f"✓ Round {round_num + 1} successful!")
        else:
            print(f"✗ Round {round_num + 1} failed (not enough clients)")

if __name__ == "__main__":
    simulate_federated_learning()
```

### Key Takeaways
- **Rounds** = Raft terms
- **Model distribution** = Log replication
- **Quorum** = Minimum clients needed for aggregation
- **Commit** = Update global model
- **Privacy** = Data never leaves clients (like Raft peers maintain local state)

---

## Example 4: Fault-Tolerant Training (Raft Checkpointing)

### Concept
Checkpoint training state like Raft persists state, enabling recovery from failures.

### Implementation

```python
import torch
import os
import hashlib
from pathlib import Path

class FaultTolerantTrainer:
    """
    Like Raft persistence (Lab 3C):
    - Checkpoint state regularly
    - Recover from crashes
    - Verify integrity
    """
    def __init__(self, model, checkpoint_dir='checkpoints', checkpoint_freq=10):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_freq = checkpoint_freq

        # Training state (like Raft persistent state)
        self.step = 0
        self.epoch = 0
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def checkpoint(self):
        """
        Persist state (like rf.persist() in Raft)

        Saves:
        - Model state (like Raft log)
        - Optimizer state (like Raft commitIndex)
        - Training progress (like Raft currentTerm)
        - RNG state (for reproducibility)
        """
        state = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{self.step}.pt'
        torch.save(state, checkpoint_path)

        # Compute checksum (integrity check)
        checksum = self._compute_checksum(checkpoint_path)

        # Save checksum
        checksum_path = self.checkpoint_dir / f'checkpoint_{self.step}.sha256'
        with open(checksum_path, 'w') as f:
            f.write(checksum)

        print(f"✓ Checkpoint saved at step {self.step}")
        return checkpoint_path

    def recover(self, checkpoint_path=None):
        """
        Recover from checkpoint (like rf.readPersist() in Raft)
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
            if not checkpoints:
                print("No checkpoint found, starting from scratch")
                return False
            checkpoint_path = checkpoints[-1]

        # Verify integrity
        if not self._verify_checkpoint(checkpoint_path):
            print(f"✗ Checkpoint {checkpoint_path} corrupted!")
            return False

        # Load state
        state = torch.load(checkpoint_path)
        self.step = state['step']
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        torch.set_rng_state(state['rng_state'])

        print(f"✓ Recovered from step {self.step}")
        return True

    def _compute_checksum(self, path):
        """Compute SHA-256 checksum for integrity"""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_checkpoint(self, checkpoint_path):
        """Verify checkpoint integrity"""
        checksum_path = checkpoint_path.with_suffix('.sha256')
        if not checksum_path.exists():
            print("Warning: No checksum found")
            return True

        expected_checksum = checksum_path.read_text().strip()
        actual_checksum = self._compute_checksum(checkpoint_path)

        return expected_checksum == actual_checksum

    def train_step(self, batch):
        """
        Single training step with automatic checkpointing
        """
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1

        # Checkpoint periodically (like Raft periodic persistence)
        if self.step % self.checkpoint_freq == 0:
            self.checkpoint()

        return loss.item()

def simulate_training_with_failure():
    """Simulate training with a crash and recovery"""
    model = SimpleModel()
    trainer = FaultTolerantTrainer(model, checkpoint_freq=5)

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=32)

    print("=== Training for 15 steps ===")
    for i, batch in enumerate(loader):
        loss = trainer.train_step(batch)
        print(f"Step {trainer.step}: loss = {loss:.4f}")

        # Simulate crash at step 12
        if trainer.step == 12:
            print("\n💥 CRASH! Simulating system failure...\n")
            break

    # Simulate recovery
    print("=== Recovering from checkpoint ===")
    new_trainer = FaultTolerantTrainer(SimpleModel(), checkpoint_freq=5)
    recovered = new_trainer.recover()

    if recovered:
        print(f"✓ Training resumed from step {new_trainer.step}")

        # Continue training
        print("\n=== Continuing training ===")
        for i, batch in enumerate(loader):
            if new_trainer.step >= 20:
                break
            loss = new_trainer.train_step(batch)
            print(f"Step {new_trainer.step}: loss = {loss:.4f}")

if __name__ == "__main__":
    simulate_training_with_failure()
```

### Key Takeaways
- **Checkpoint** = Raft persist()
- **Recovery** = Raft readPersist()
- **State to save** = Model, optimizer, training progress (like Raft's currentTerm, log, votedFor)
- **Integrity checks** = Ensure checkpoint not corrupted
- **Periodic saves** = Balance between overhead and data loss

---

## Running the Examples

### Prerequisites
```bash
pip install torch torchvision
```

### Run Individual Examples
```bash
# Example 1: Data Parallelism
python practical_examples.py --example=data_parallel

# Example 2: Parameter Server
python practical_examples.py --example=param_server

# Example 3: Federated Learning
python practical_examples.py --example=federated

# Example 4: Fault Tolerance
python practical_examples.py --example=fault_tolerant
```

### Performance Comparison

Run benchmarks to see the trade-offs:
```bash
python benchmark.py --workers=1,2,4,8 --sync=true,false
```

---

## Exercises

### Exercise 1: Add Worker Failure Handling
Extend Example 1 (Data Parallel) to detect and handle worker failures:
- Timeout for slow workers
- Reassign tasks to other workers
- Compare to Lab 1 MapReduce coordinator

### Exercise 2: Implement Stale-Synchronous Parallelism
Extend Example 2 (Parameter Server) to support bounded staleness:
- Track version numbers per worker
- Allow workers to be ahead by at most K steps
- Measure impact on convergence

### Exercise 3: Add Byzantine Fault Tolerance
Extend Example 3 (Federated Learning) to handle malicious clients:
- Detect outlier gradients
- Use median instead of mean for aggregation
- Compare to Raft's assumption of crash-only failures

### Exercise 4: Implement Model Sharding
Create a new example for model parallelism (Lab 5 style):
- Partition model layers across "GPUs" (simulated)
- Pass activations between partitions
- Support dynamic reconfiguration

---

## Next Steps

1. **Experiment**: Modify the examples, break things, fix them
2. **Measure**: Add timing, communication overhead metrics
3. **Scale**: Try with real datasets (MNIST, CIFAR-10)
4. **Compare**: Benchmark different strategies
5. **Research**: Read papers in `distributed-systems-for-ai-research.md`

Happy experimenting!
