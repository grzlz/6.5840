# Distributed Systems Concepts for AI Research

## Introduction

Modern AI research increasingly relies on distributed systems principles to train, serve, and scale large models. This lesson explores how the fundamental concepts from MIT 6.5840 (Distributed Systems) directly apply to cutting-edge AI research and infrastructure.

**Key Insight**: The challenges solved in distributed systems—coordination, fault tolerance, scalability, consistency—are the same challenges facing modern AI at scale.

---

## Table of Contents

1. [MapReduce → Data Parallelism & Large-Scale Training](#1-mapreduce--data-parallelism--large-scale-training)
2. [Raft Consensus → Federated Learning & Coordination](#2-raft-consensus--federated-learning--coordination)
3. [Sharding → Model Parallelism & Partitioning](#3-sharding--model-parallelism--partitioning)
4. [Fault Tolerance → Resilient AI Training](#4-fault-tolerance--resilient-ai-training)
5. [Replication → Model Serving & Inference](#5-replication--model-serving--inference)
6. [Consistency Models → Distributed Gradient Descent](#6-consistency-models--distributed-gradient-descent)
7. [Real-World AI Systems](#7-real-world-ai-systems)
8. [Research Opportunities](#8-research-opportunities)

---

## 1. MapReduce → Data Parallelism & Large-Scale Training

### The Connection

**MapReduce (Lab 1)** introduced the paradigm of:
- Distributing computation across many workers
- Processing data in parallel
- Coordinating results through a central coordinator

**In AI**: This is precisely how we train large models on massive datasets.

### Data Parallelism in Neural Networks

```
Traditional MapReduce:
┌──────────────┐
│ Coordinator  │ ← Distributes tasks, collects results
└──────────────┘
       ↓
┌────────┬────────┬────────┐
│Worker 1│Worker 2│Worker 3│ ← Process data chunks
└────────┴────────┴────────┘

Modern AI Training:
┌──────────────────┐
│ Parameter Server │ ← Stores model, aggregates gradients
└──────────────────┘
       ↓
┌────────┬────────┬────────┐
│ GPU 1  │ GPU 2  │ GPU 3  │ ← Train on data batches
└────────┴────────┴────────┘
```

### Key Parallels

| MapReduce Concept | AI Training Equivalent |
|------------------|------------------------|
| Input splits | Data batches/shards |
| Map function | Forward pass + gradient computation |
| Shuffle & Sort | Gradient aggregation |
| Reduce function | Parameter update |
| Coordinator | Parameter server / All-Reduce coordinator |
| Worker failure handling | GPU failure recovery, checkpoint restart |

### Example: Training GPT-Style Models

**Traditional (Single GPU):**
```
Time to train GPT-3 (175B params): ~355 GPU-years on V100
```

**With Data Parallelism (MapReduce-style):**
```
1,024 GPUs → ~127 days
10,240 GPUs → ~12.7 days (with perfect scaling)
```

### Code Pattern Comparison

**MapReduce Coordinator (Lab 1):**
```go
type Coordinator struct {
    tasks    []Task
    workers  map[int]*Worker
    results  chan Result
}

func (c *Coordinator) DistributeTasks() {
    for _, task := range c.tasks {
        worker := c.selectWorker()
        worker.Execute(task)
    }
}
```

**AI Parameter Server Pattern:**
```python
class ParameterServer:
    def __init__(self, model):
        self.params = model.parameters()
        self.workers = []
        self.gradient_queue = Queue()

    def distribute_batches(self, dataset):
        for batch in dataset.shard(num_workers):
            worker = self.select_worker()
            worker.compute_gradients(batch, self.params)

    def aggregate_gradients(self):
        gradients = [self.gradient_queue.get()
                     for _ in self.workers]
        return sum(gradients) / len(gradients)
```

### Research Applications

1. **ImageNet Training**: Distribute 1.2M images across workers
2. **Language Model Pretraining**: Process billions of tokens in parallel
3. **Reinforcement Learning**: Parallel environment simulation (e.g., AlphaGo)

### Challenges from MapReduce Applied to AI

- **Straggler handling**: Slow GPUs delay entire batch → Dynamic batch sizing
- **Communication overhead**: Gradient aggregation can bottleneck → Gradient compression
- **Fault tolerance**: GPU failures → Checkpoint/restart mechanisms

---

## 2. Raft Consensus → Federated Learning & Coordination

### The Connection

**Raft (Lab 3)** solves the fundamental problem:
> How do independent machines agree on a shared state in the presence of failures?

**In AI**: This is critical for:
- Federated learning across edge devices
- Distributed hyperparameter tuning
- Multi-agent reinforcement learning
- Coordinating training across data centers

### Federated Learning as Consensus

**Federated Learning Setup:**
```
Hospital 1 ──┐
Hospital 2 ──┼──→ Coordinator ──→ Global Model
Hospital 3 ──┘
    ↑
    └── Local data cannot leave premises (privacy)
```

**Raft-Inspired Solution:**
- Each hospital = Raft peer
- Model updates = Log entries
- Leader = Aggregation coordinator
- Consensus = Agreement on which updates to apply

### Key Parallels

| Raft Concept | AI Equivalent |
|--------------|---------------|
| Leader election | Coordinator selection for aggregation |
| Log replication | Distributing model updates |
| Committed entries | Accepted global model updates |
| Snapshots | Model checkpoints |
| Heartbeats | Liveness checks for workers |
| Term numbers | Training rounds/epochs |

### Example: Federated Learning Protocol

**Raft Log Entry:**
```go
type LogEntry struct {
    Term    int
    Command interface{}
    Index   int
}
```

**Federated Learning Round:**
```python
class FederatedRound:
    def __init__(self, round_num, model_updates, participants):
        self.round = round_num           # Like Raft term
        self.updates = model_updates     # Like command
        self.participants = participants # Like voting peers

    def commit_if_quorum(self):
        # Similar to Raft commit rule
        if len(self.updates) >= len(self.participants) // 2 + 1:
            return self.aggregate_updates()
        return None
```

### Byzantine Fault Tolerance for AI

Standard Raft assumes **crash faults**. But in AI:

**Problem**: Malicious participants might send poisoned gradients

**Solution**: Byzantine fault-tolerant aggregation
```python
def byzantine_robust_aggregate(gradients):
    """
    Like Raft consensus, but tolerating malicious updates
    """
    # Remove outliers (potential attacks)
    median_gradients = geometric_median(gradients)

    # Only accept if majority agree (like Raft quorum)
    if agreement_score(gradients) > threshold:
        return median_gradients
    return None  # No consensus
```

### Research Applications

1. **Cross-Silo Federated Learning**: Hospitals, banks coordinate model training
2. **Edge AI**: Mobile devices collaboratively train models
3. **Multi-Agent RL**: Agents agree on shared policy
4. **Distributed AutoML**: Coordinate hyperparameter search across clusters

### Case Study: Google Gboard

**Challenge**: Train next-word prediction across millions of phones

**Solution**: Federated learning with Raft-like coordination
```
1. Coordinator selects participants (like Raft leader election)
2. Send current model to devices (like log replication)
3. Devices compute updates locally
4. Aggregate updates at coordinator (like commit)
5. Apply update if quorum reached
6. Repeat (next term/round)
```

**Results**:
- Privacy preserved (data never leaves devices)
- Model quality competitive with centralized training
- Resilient to device failures (Raft fault tolerance)

---

## 3. Sharding → Model Parallelism & Partitioning

### The Connection

**Sharding (Lab 5)** partitions data across multiple replica groups:
- Each shard handles a portion of the keyspace
- Load balancing across shards
- Dynamic reconfiguration as shards join/leave

**In AI**: This is exactly how we partition large models that don't fit on a single GPU.

### Model Parallelism Architectures

**Pipeline Parallelism: Sharding by Layers**
```
┌─────────────┐
│   GPU 1     │ ← Layers 1-4 (Shard 1)
└──────┬──────┘
       ↓
┌─────────────┐
│   GPU 2     │ ← Layers 5-8 (Shard 2)
└──────┬──────┘
       ↓
┌─────────────┐
│   GPU 3     │ ← Layers 9-12 (Shard 3)
└─────────────┘
```

**Tensor Parallelism: Sharding within Layers**
```
Attention Layer (sharded across GPUs)
┌──────────┬──────────┬──────────┐
│ GPU 1    │ GPU 2    │ GPU 3    │
│ Heads    │ Heads    │ Heads    │
│ 1-4      │ 5-8      │ 9-12     │
└──────────┴──────────┴──────────┘
```

### Key Parallels

| Sharding Concept | AI Parallel |
|------------------|-------------|
| Shard = keyspace partition | Model partition (layers/tensors) |
| Shard controller | Orchestrator (e.g., Megatron scheduler) |
| Data migration | Activation passing between GPUs |
| Rebalancing | Load balancing across devices |
| Configuration changes | Dynamic model scaling |

### ShardKV → Model Serving Architecture

**Lab 5 Sharded KV:**
```
Shard Group 1 (Keys A-M)  ───┐
Shard Group 2 (Keys N-Z)  ───┼──→ Client Router
Shard Group 3 (Overflow)  ───┘
```

**Sharded Model Serving:**
```
Model Shard 1 (Users A-M)  ───┐
Model Shard 2 (Users N-Z)  ───┼──→ Load Balancer
Model Shard 3 (Overflow)   ───┘
```

### Example: GPT-3 Model Partitioning

**Problem**: GPT-3 (175B parameters) = ~350 GB in FP16
- Single A100 GPU = 80 GB memory
- Need: 350 / 80 ≈ 5 GPUs minimum

**Solution: 3D Parallelism (Sharding Strategy)**
```python
class ModelPartitioner:
    def __init__(self, model, num_gpus):
        self.model = model
        self.num_gpus = num_gpus

    def partition_pipeline(self, num_stages):
        """
        Like sharding by keyspace:
        Shard 1 → Layers 0-23
        Shard 2 → Layers 24-47
        ...
        """
        layers_per_stage = len(self.model.layers) // num_stages
        for stage in range(num_stages):
            start = stage * layers_per_stage
            end = start + layers_per_stage
            assign_to_gpu(self.model.layers[start:end], gpu=stage)

    def partition_tensor(self, tensor, num_shards):
        """
        Like sharding within a shard:
        Split attention heads across GPUs
        """
        return torch.chunk(tensor, num_shards, dim=-1)

    def partition_data(self, batch, num_replicas):
        """
        Like replication in Lab 5:
        Each replica processes different data
        """
        return torch.chunk(batch, num_replicas, dim=0)
```

### Shard Migration in AI

**Lab 5 Challenge**: Move data between shard groups during reconfiguration

**AI Equivalent**: Transfer model partitions during training
```python
def dynamic_rebalancing(model_shards, gpu_utilization):
    """
    Like Lab 5 shard migration:
    Rebalance model partitions based on GPU load
    """
    overloaded_gpus = [g for g in gpus if utilization[g] > 0.9]
    underutilized_gpus = [g for g in gpus if utilization[g] < 0.5]

    for src_gpu in overloaded_gpus:
        for dst_gpu in underutilized_gpus:
            # Migrate shard (like Lab 5)
            layer = select_layer_to_migrate(src_gpu)
            transfer_state(layer, src_gpu, dst_gpu)
            update_routing(layer, dst_gpu)
```

### Research Applications

1. **Megatron-LM** (NVIDIA): 3D parallelism for trillion-parameter models
2. **DeepSpeed** (Microsoft): ZeRO sharding to reduce memory
3. **Switch Transformers**: Sparse sharding with MoE (Mixture of Experts)
4. **Recommendation Systems**: Embedding table sharding

### Case Study: Training 1 Trillion Parameter Models

**GPT-3**: 175B params, 10,000 GPUs
**Megatron-Turing NLG**: 530B params, uses 3D sharding

**Sharding Strategy:**
```
Data Parallel: 64 ways (different batches)
Pipeline Parallel: 8 stages (layer sharding)
Tensor Parallel: 8 ways (within-layer sharding)
Total: 64 × 8 × 8 = 4,096 GPUs
```

**Lab 5 Lessons Applied:**
- Shard controller → Orchestrator coordinates all partitions
- Configuration changes → Can add/remove GPUs dynamically
- Data migration → Checkpoint/restart with different topology

---

## 4. Fault Tolerance → Resilient AI Training

### The Connection

**Fault Tolerance (Throughout Labs 1-5)**:
- MapReduce: Worker failure detection and task reassignment
- Raft: Survive minority of failures via replication
- KVRaft: Persist state to survive crashes
- ShardKV: Reconfigure when replica groups fail

**In AI**: Training runs can last weeks/months and cost millions of dollars. Fault tolerance is not optional.

### AI Training Failure Modes

```
┌─────────────────────┬──────────────────┬─────────────────┐
│ Failure Type        │ Frequency        │ Impact          │
├─────────────────────┼──────────────────┼─────────────────┤
│ GPU Hardware Fail   │ ~1-2% daily*     │ Job crash       │
│ Network Partition   │ ~0.1% daily      │ Stragglers      │
│ OOM (Memory)        │ 10-20% of runs   │ Immediate crash │
│ Software Bug        │ 5-10% of runs    │ Silent errors   │
│ Power Outage        │ Rare but happens │ Complete loss   │
└─────────────────────┴──────────────────┴─────────────────┘
* At scale (1000+ GPUs)
```

### Raft-Style Checkpointing for AI

**Raft Persistence (Lab 3C):**
```go
func (rf *Raft) persist() {
    w := new(bytes.Buffer)
    e := labgob.NewEncoder(w)
    e.Encode(rf.currentTerm)
    e.Encode(rf.votedFor)
    e.Encode(rf.log)
    rf.persister.Save(w.Bytes(), nil)
}
```

**AI Model Checkpointing:**
```python
class FaultTolerantTrainer:
    def __init__(self, model, checkpoint_freq=100):
        self.model = model
        self.checkpoint_freq = checkpoint_freq
        self.step = 0

    def train_step(self, batch):
        # Compute loss and gradients
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()

        self.step += 1

        # Persist state (like Raft)
        if self.step % self.checkpoint_freq == 0:
            self.checkpoint()

    def checkpoint(self):
        """Like rf.persist() in Raft"""
        state = {
            'step': self.step,              # Like currentTerm
            'model': self.model.state_dict(),  # Like log
            'optimizer': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
        }
        torch.save(state, f'checkpoint_{self.step}.pt')

    def recover(self, checkpoint_path):
        """Like rf.readPersist() in Raft"""
        state = torch.load(checkpoint_path)
        self.step = state['step']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_state'])
```

### Replication for Fault Tolerance

**Raft Approach**: Replicate state across multiple servers

**AI Training Approach**: Redundant computation
```python
def replicated_training(model, data, num_replicas=3):
    """
    Like Raft replication:
    Multiple workers compute same gradients
    """
    results = []
    for replica in range(num_replicas):
        grad = compute_gradient_on_replica(model, data, replica)
        results.append(grad)

    # Consensus: majority vote (like Raft commit)
    if all_agree(results):
        return results[0]
    else:
        # Detect Byzantine failure
        return median(results)
```

### Elastic Training (Like ShardKV Reconfiguration)

**Problem**: GPU fails mid-training, how to continue?

**Lab 5 Solution**: Reconfigure shard groups when replicas fail

**AI Solution**: Elastic training
```python
class ElasticTrainer:
    def __init__(self, model, min_workers=4, max_workers=16):
        self.model = model
        self.workers = []
        self.min_workers = min_workers

    def on_worker_failure(self, failed_worker):
        """Like Lab 5 reconfiguration"""
        self.workers.remove(failed_worker)

        if len(self.workers) < self.min_workers:
            # Request new worker (like shard migration)
            new_worker = self.request_worker()
            self.workers.append(new_worker)

            # Transfer state to new worker
            self.sync_worker(new_worker, self.latest_checkpoint)

    def on_worker_join(self, new_worker):
        """Scale up when resources available"""
        if len(self.workers) < self.max_workers:
            self.workers.append(new_worker)
            self.rebalance_work()  # Like shard rebalancing
```

### Research Applications

1. **PyTorch Elastic**: Automatic recovery from failures
2. **TensorFlow Fault Tolerance**: Checkpoint/restore mechanisms
3. **Ray**: Distributed training with automatic failure recovery
4. **Horovod**: Elastic training for dynamic resource allocation

### Case Study: Meta's 175B Parameter Model

**Challenge**: Training for 3 months on 1,024 GPUs
- Probability of 0 failures ≈ 0% (essentially guaranteed failures)

**Solution Stack (Inspired by 6.5840)**:
```
┌──────────────────────────────────────┐
│ Raft-style Checkpointing             │ ← Lab 3C
│ - Checkpoint every 100 steps         │
│ - Async checkpoint to cloud storage  │
├──────────────────────────────────────┤
│ MapReduce-style Worker Management    │ ← Lab 1
│ - Detect stragglers                  │
│ - Reassign work to healthy GPUs      │
├──────────────────────────────────────┤
│ ShardKV-style Reconfiguration        │ ← Lab 5
│ - Add/remove GPUs dynamically        │
│ - Rebalance model shards             │
└──────────────────────────────────────┘
```

**Results**:
- 47 GPU failures during training
- Average recovery time: 8 minutes
- Total wasted compute: <0.5%

---

## 5. Replication → Model Serving & Inference

### The Connection

**Replication in Raft (Lab 3)**:
- Multiple servers hold copies of the same state
- Serve read requests from any replica
- Provides high availability and load balancing

**In AI Serving**:
- Multiple instances of the same model
- Serve inference requests from any instance
- Scale to handle millions of queries per second

### Model Serving Architecture

**Raft Cluster:**
```
Client ──→ Leader ──→ Apply to state machine
          ↓
       Follower 1 ──→ Serve reads
       Follower 2 ──→ Serve reads
       Follower 3 ──→ Serve reads
```

**AI Model Serving:**
```
User Request ──→ Load Balancer
                    ↓
                 ┌──┼──┐
          Replica 1│ 2 │3  ──→ Return predictions
          ┌────────┴───┴─────┐
          │  Same Model Copy  │
          └───────────────────┘
```

### Key Parallels

| Raft Concept | Model Serving |
|--------------|---------------|
| State machine replication | Model replication |
| Read from followers | Inference from any replica |
| Leader for writes | Training cluster |
| Log = sequence of ops | Request queue |
| Snapshot | Model checkpoint |
| Quorum reads | Ensemble inference |

### Consistency in Model Serving

**Lab 3 Linearizability**: All clients see same state

**AI Serving Challenges**:
```python
class ModelServingCluster:
    def __init__(self, model_path, num_replicas=5):
        self.replicas = [
            load_model(model_path) for _ in range(num_replicas)
        ]
        self.version = 0

    def predict(self, input_data):
        """
        Like Raft read:
        Can serve from any replica
        """
        replica = random.choice(self.replicas)
        return replica(input_data)

    def update_model(self, new_model_path):
        """
        Like Raft log replication:
        Need to update all replicas consistently
        """
        self.version += 1

        # Rolling update (like Raft log replication)
        for replica in self.replicas:
            new_model = load_model(new_model_path)
            # Atomic swap
            replica.swap(new_model)
            # Health check
            assert replica.version == self.version
```

### Read-Only Replicas vs. Training

**Raft Design**:
- Leader handles writes (consensus-critical)
- Followers can serve reads (optimization)

**AI Design**:
- Training cluster handles updates (expensive)
- Serving replicas handle inference (cheap, parallelizable)

```python
class SeparatedTrainingServing:
    def __init__(self):
        # Like Raft leader
        self.training_cluster = TrainingCluster(gpus=1024)

        # Like Raft followers
        self.serving_replicas = [
            InferenceEngine(gpu=i) for i in range(100)
        ]

    def train_new_version(self, dataset):
        """Like Raft leader accepting writes"""
        new_model = self.training_cluster.train(dataset)

        # Replicate to serving (like log replication)
        self.deploy_to_serving(new_model)

    def deploy_to_serving(self, model):
        """Like Raft AppendEntries RPC"""
        for replica in self.serving_replicas:
            replica.load_model(model)
            # Wait for acknowledgment
            assert replica.health_check()
```

### Ensemble Models as Quorum Reads

**Raft Quorum Reads**: Read from majority to ensure fresh data

**AI Ensemble**: Query multiple models for better predictions

```python
def ensemble_inference(models, input_data, require_quorum=True):
    """
    Like Raft quorum read:
    Get predictions from multiple replicas
    """
    predictions = []
    for model in models:
        pred = model(input_data)
        predictions.append(pred)

    if require_quorum:
        # Need majority agreement (like Raft)
        return majority_vote(predictions)
    else:
        # Average all predictions
        return average(predictions)
```

### Research Applications

1. **TensorFlow Serving**: Multi-version model serving
2. **TorchServe**: Scalable model deployment
3. **Triton Inference Server**: GPU-accelerated serving
4. **KServe**: Kubernetes-native model serving

### Case Study: Google Search Autocomplete

**Requirements**:
- Serve 3.5 billion searches/day
- Latency < 100ms
- 99.99% availability

**Solution (Raft-inspired)**:
```
┌──────────────────────────────────────┐
│ Global Load Balancer                 │
└────────────┬─────────────────────────┘
             ↓
┌────────────┴─────────────────────────┐
│ Regional Clusters (Like Raft Groups) │
├──────────────────────────────────────┤
│ US-East:  100 replicas               │
│ US-West:  100 replicas               │
│ Europe:   80 replicas                │
│ Asia:     120 replicas               │
└──────────────────────────────────────┘
```

**Raft Lessons Applied**:
- Replication → Multiple model instances
- Fault tolerance → Replica failures don't affect service
- Leader election → If region fails, route to another
- Snapshots → Model updates deployed gradually

---

## 6. Consistency Models → Distributed Gradient Descent

### The Connection

**Consistency in Distributed Systems (Throughout Labs)**:
- Linearizability: Operations appear atomic and ordered
- Eventual consistency: Updates eventually propagate
- Strong consistency: All nodes see same state

**In Distributed AI Training**:
- How do we coordinate gradient updates across workers?
- When is the global model "consistent"?
- Trade-offs between speed and accuracy

### Consistency Spectrum in AI Training

```
Strong Consistency                     Weak Consistency
(Slow, Accurate)                      (Fast, Less Accurate)
      ↓                                       ↓
Synchronous SGD ←→ Stale Sync SGD ←→ Async SGD
   (BSP)              (SSP)            (ASP)
```

### Synchronous Training (Linearizable)

**Like Raft**: All updates must reach consensus before proceeding

```python
class SynchronousSGD:
    """
    Strong consistency: Like Raft linearizability
    All workers must complete before next step
    """
    def __init__(self, model, workers):
        self.model = model
        self.workers = workers
        self.barrier = Barrier(len(workers))

    def train_step(self, data_shards):
        # Each worker computes gradients
        gradients = []
        for worker, data in zip(self.workers, data_shards):
            grad = worker.compute_gradient(self.model, data)
            gradients.append(grad)

        # Barrier: wait for ALL workers (like Raft quorum)
        self.barrier.wait()

        # Aggregate gradients (like Raft commit)
        avg_gradient = sum(gradients) / len(gradients)

        # Apply to model (like Raft apply to state machine)
        self.model.update(avg_gradient)

        # Next iteration starts with consistent model
```

**Pros**: Identical to single-GPU training (mathematically equivalent)
**Cons**: Slow workers block everyone (straggler problem)

### Asynchronous Training (Eventual Consistency)

**Like Eventual Consistency**: Workers update independently

```python
class AsynchronousSGD:
    """
    Weak consistency: Like eventual consistency
    Workers update parameter server independently
    """
    def __init__(self, model):
        self.params = ParameterServer(model)
        self.workers = []

    def worker_loop(self, worker_id, data):
        while True:
            # 1. Pull latest params (might be stale!)
            local_params = self.params.pull()

            # 2. Compute gradient locally
            gradient = compute_gradient(local_params, data.next_batch())

            # 3. Push gradient (no waiting for others!)
            self.params.push(gradient, worker_id)

            # Workers run at different speeds
            # Parameter server sees inconsistent updates
```

**Pros**: No waiting for stragglers (fast)
**Cons**: Stale gradients can hurt convergence

### Stale-Synchronous Parallelism (SSP)

**Bounded Staleness**: Middle ground between sync and async

```python
class StaleSynchronousSGD:
    """
    Bounded staleness: Like bounded eventual consistency
    Allow some lag but not unbounded
    """
    def __init__(self, model, workers, staleness_bound=3):
        self.params = ParameterServer(model)
        self.workers = workers
        self.staleness = staleness_bound
        self.worker_clocks = [0] * len(workers)

    def worker_loop(self, worker_id, data):
        while True:
            # Check staleness
            min_clock = min(self.worker_clocks)
            my_clock = self.worker_clocks[worker_id]

            # If too far ahead, wait (like Raft log lag)
            if my_clock - min_clock >= self.staleness:
                wait_for_slowest_worker()

            # Proceed with update
            local_params = self.params.pull()
            gradient = compute_gradient(local_params, data.next_batch())
            self.params.push(gradient, worker_id)

            self.worker_clocks[worker_id] += 1
```

**Pros**: Balance speed and consistency
**Cons**: Still some convergence degradation

### Consistency in Parameter Server Architecture

**Lab 2 KV Server** → **Parameter Server**

```python
# Lab 2 Style
class KVServer:
    def Get(self, key):
        return self.data[key]

    def Put(self, key, value):
        self.data[key] = value

# AI Parameter Server (same interface!)
class ParameterServer:
    def Pull(self, layer_name):
        """Like Get: retrieve parameters"""
        return self.parameters[layer_name]

    def Push(self, layer_name, gradient):
        """Like Put: update parameters"""
        self.parameters[layer_name] -= lr * gradient
        self.version[layer_name] += 1

    def PullWithVersion(self, layer_name):
        """Like linearizable read in Lab 2"""
        return {
            'params': self.parameters[layer_name],
            'version': self.version[layer_name]
        }
```

### Consistency Guarantees Comparison

| System | Guarantee | Perf | Use Case |
|--------|-----------|------|----------|
| Raft | Linearizable | Slow | Critical state |
| Lab 2 KV | Linearizable | Medium | Correctness critical |
| Sync SGD | Exact gradients | Slow | Research, small scale |
| SSP | Bounded stale | Medium | Production training |
| Async SGD | Eventually consistent | Fast | Large-scale, robust models |

### Research Applications

1. **Horovod**: Ring-AllReduce for synchronous training
2. **BytePS**: Parameter server with bounded staleness
3. **PyTorch DDP**: Synchronous distributed training
4. **DistBelief** (Google): Asynchronous parameter server

### Case Study: Training ResNet-50 on ImageNet

**Baseline (Single GPU)**: 14 days

**Synchronous (256 GPUs)**:
```
Speedup: 220x (not 256x due to communication)
Time: 1.5 hours
Accuracy: Same as single GPU
```

**Asynchronous (256 GPUs)**:
```
Speedup: 240x (better efficiency)
Time: 1.4 hours
Accuracy: -0.5% (slight degradation)
```

**Stale-Synchronous (256 GPUs, staleness=4)**:
```
Speedup: 235x (compromise)
Time: 1.43 hours
Accuracy: -0.1% (minimal degradation)
```

### Implementing Consistency Checks

```python
def verify_consistency(parameter_server, workers):
    """
    Like Lab 2 consistency checks:
    Verify all workers have consistent view
    """
    versions = []
    for worker in workers:
        v = worker.get_param_version()
        versions.append(v)

    # Strong consistency (like linearizability)
    assert all(v == versions[0] for v in versions), \
        "Consistency violation: workers have different versions"

    # Or check bounded staleness (like SSP)
    assert max(versions) - min(versions) <= STALENESS_BOUND, \
        "Staleness bound violated"
```

---

## 7. Real-World AI Systems

### How Major AI Labs Use Distributed Systems

#### OpenAI GPT-3/GPT-4

**Architecture Stack**:
```
┌─────────────────────────────────────────┐
│ ShardKV-style serving (Lab 5)           │ ← Inference API
├─────────────────────────────────────────┤
│ Megatron (Model Parallelism)            │ ← Lab 5 sharding
├─────────────────────────────────────────┤
│ Data Parallelism                        │ ← Lab 1 MapReduce
├─────────────────────────────────────────┤
│ Fault-tolerant checkpointing            │ ← Lab 3 persistence
└─────────────────────────────────────────┘
```

**Key Techniques from 6.5840**:
- MapReduce-style data parallelism (Lab 1)
- Raft-style checkpointing every N steps (Lab 3C)
- 3D sharding of model (Lab 5)
- Replicated serving (Lab 3 replication)

#### Google BERT/T5/PaLM

**Infrastructure**:
```
Cloud TPU Pods (4096 TPU cores)
├── JAX/Pax framework
│   ├── GSPMD: Auto-sharding (Lab 5 concepts)
│   ├── Async checkpointing (Lab 3C)
│   └── Data parallelism (Lab 1)
└── Pathways: Multi-model serving
    └── Replicated inference (Lab 3)
```

#### Meta PyTorch FSDP (Fully Sharded Data Parallel)

**Design**:
```python
class FSDP:
    """
    Combines all 6.5840 labs:
    - Data parallelism (Lab 1)
    - Sharding parameters (Lab 5)
    - Fault tolerance (Lab 3)
    """
    def __init__(self, model):
        # Shard model across GPUs (Lab 5)
        self.sharded_params = self.shard_parameters(model)

        # Each GPU is a replica group
        self.replica_group = init_process_group()

    def forward(self, input):
        # All-gather parameters (like Lab 5 shard query)
        full_params = all_gather(self.sharded_params)

        # Forward pass
        output = self.model(input, full_params)

        # Free memory immediately
        del full_params
        return output

    def backward(self, loss):
        # Compute gradients
        gradients = loss.backward()

        # Reduce-scatter (like MapReduce)
        self.sharded_grads = reduce_scatter(gradients)

        # Update only local shard
        self.sharded_params -= lr * self.sharded_grads
```

#### DeepMind AlphaFold/AlphaStar

**Multi-Agent RL** (AlphaStar):
```
League Training:
├── Coordinator (like MapReduce coordinator)
│   ├── Assigns opponents
│   └── Collects game results
├── Self-play workers (like MR workers)
│   ├── 600+ instances
│   └── Play games, collect trajectories
└── Training cluster
    ├── Parameter server (like Lab 2 KV)
    └── Aggregates gradients from workers
```

### System Design Patterns

**Pattern 1: Parameter Server** (from Lab 2 + Lab 3)
```
Used by: DistBelief (Google), Petuum, Angel (Tencent)

┌──────────────┐
│  PS Leader   │ ← Like Raft leader
└──────┬───────┘
       ↓
┌──────┴───────┐
│ PS Replicas  │ ← Like Raft followers
└──────────────┘
       ↑
┌──────┴───────┐
│   Workers    │ ← Push/pull gradients
└──────────────┘
```

**Pattern 2: Ring AllReduce** (from Lab 1 + P2P)
```
Used by: Horovod, PyTorch DDP, NCCL

Worker 1 ←→ Worker 2 ←→ Worker 3 ←→ Worker 4
    ↑                                    ↓
    └────────────────────────────────────┘

Each worker sends chunk to next in ring
No central coordinator needed
```

**Pattern 3: Hierarchical AllReduce**
```
Used by: BytePS, Large-scale training

         Root Aggregator
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
 Rack 1    Rack 2    Rack 3
  (8 GPU)  (8 GPU)  (8 GPU)

Within rack: fast AllReduce
Across racks: slower network
```

---

## 8. Research Opportunities

### Open Problems at Intersection of Distributed Systems × AI

#### 1. Federated Learning Under Byzantine Faults

**Problem**: Raft assumes crash faults. Federated learning has malicious clients.

**Research Questions**:
- Can we adapt BFT consensus for gradient aggregation?
- How to detect poisoned gradients efficiently?
- Trade-offs between Byzantine tolerance and communication cost?

**Starting Point**:
```python
# Extend Lab 3 Raft with Byzantine detection
class ByzantineRobustFederated:
    def aggregate_gradients(self, gradients, signatures):
        # Verify signatures (authenticity)
        verified = [g for g, sig in zip(gradients, signatures)
                    if verify(g, sig)]

        # Byzantine filtering (geometric median)
        # Need >2f+1 honest workers (like BFT)
        if len(verified) >= 2 * self.f + 1:
            return geometric_median(verified)

        return None  # Cannot reach consensus
```

#### 2. Efficient Model Serving with Consistency

**Problem**: Serving models while training updates them. How to ensure consistency?

**Research Questions**:
- Like Raft linearizable reads, can we have linearizable inference?
- How to version models during continuous training?
- Shadow deployment strategies?

**Connection to Lab 3**:
```python
# Extend KVRaft (Lab 4) for model serving
class ModelServingKV:
    def __init__(self):
        self.raft = Raft()  # Consensus for versions
        self.models = {}    # Versioned models

    def serve_inference(self, input, consistency_level):
        if consistency_level == "linearizable":
            # Like Raft linearizable read
            # Must contact leader
            version = self.raft.read_latest_version()
        elif consistency_level == "stale":
            # Read from local cache (faster)
            version = self.local_version

        model = self.models[version]
        return model(input)
```

#### 3. Adaptive Sharding for Dynamic Models

**Problem**: Models change architecture during training (NAS, pruning). How to reshard?

**Research Questions**:
- Extend Lab 5 reconfiguration to handle model architecture changes
- Minimize data movement during resharding
- Online vs. offline resharding trade-offs

**Extension of Lab 5**:
```python
class AdaptiveModelSharding:
    def __init__(self, model):
        self.shard_controller = ShardController()  # Like Lab 5
        self.current_config = self.compute_sharding(model)

    def on_architecture_change(self, new_model):
        """
        Like Lab 5 reconfiguration:
        When model changes, reshard optimally
        """
        new_config = self.compute_sharding(new_model)

        # Minimize migration (like Lab 5 challenge)
        migration_plan = self.plan_migration(
            self.current_config,
            new_config
        )

        # Execute migration
        self.execute_migration(migration_plan)
        self.current_config = new_config
```

#### 4. Consistency-Aware AutoML

**Problem**: Hyperparameter search runs many experiments. How to coordinate efficiently?

**Research Questions**:
- Use Raft to coordinate distributed hyperparameter search?
- Early stopping of unpromising trials (like MapReduce speculative execution)
- Fault tolerance when experiments fail

**Combining Lab 1 + Lab 3**:
```python
class DistributedAutoML:
    def __init__(self, search_space):
        self.coordinator = Raft()  # Lab 3 for coordination
        self.workers = []          # Lab 1 style workers
        self.results = {}

    def search(self, budget):
        # Like MapReduce task distribution
        configs = self.sample_configs(budget)

        for config in configs:
            worker = self.select_worker()

            # Like Raft log replication
            # Ensure all workers know about this trial
            self.coordinator.replicate_trial(config)

            # Execute trial
            worker.train(config)

    def on_worker_result(self, config, score):
        # Like Raft commit
        self.coordinator.commit_result(config, score)

        # Early stopping (like speculative execution)
        if self.should_stop_early(config, score):
            self.cancel_trial(config)
```

#### 5. Privacy-Preserving Distributed Training

**Problem**: Train models without revealing raw data or gradients.

**Research Questions**:
- Secure multi-party computation for gradient aggregation
- Differential privacy in distributed setting
- Trade-offs between privacy and convergence

**Extending Federated Learning**:
```python
class PrivacyPreservingFL:
    def __init__(self, epsilon):  # Differential privacy budget
        self.epsilon = epsilon
        self.secure_aggregator = SecureMPC()

    def aggregate_with_privacy(self, gradients):
        # Add noise for differential privacy
        noisy_grads = [g + laplace_noise(self.epsilon)
                       for g in gradients]

        # Secure aggregation (no server sees individual gradients)
        # Like Byzantine agreement but with cryptography
        encrypted = [self.encrypt(g) for g in noisy_grads]
        aggregated_encrypted = self.secure_aggregator.sum(encrypted)

        # Only final result is decrypted
        return self.decrypt(aggregated_encrypted)
```

### Promising Research Directions

1. **Consensus for AI**
   - Lightweight consensus protocols for AI workloads
   - Relaxed consistency models that maintain convergence
   - Adaptive consistency based on training phase

2. **Fault Tolerance Beyond Checkpointing**
   - Redundant computation for gradient verification
   - Predictive failure detection using model metrics
   - Erasure coding for distributed models

3. **Sharding Algorithms**
   - Optimal sharding for heterogeneous GPUs
   - Topology-aware sharding (network-sensitive)
   - Dynamic resharding during training

4. **Distributed Debugging**
   - Deterministic replay for distributed training
   - Causality tracking across workers
   - Distributed profiling and bottleneck detection

---

## Practical Exercises

### Exercise 1: Implement Data-Parallel Training (MapReduce Style)

Build a simple data-parallel trainer using Lab 1 concepts:

```python
class SimpleDataParallel:
    """
    Implement MapReduce-style data parallelism

    Map: Each worker computes gradients on a batch
    Reduce: Aggregate gradients and update model
    """
    def __init__(self, model, num_workers=4):
        self.model = model
        self.workers = [Worker(i) for i in range(num_workers)]

    # TODO: Implement train_epoch using MapReduce pattern
    def train_epoch(self, dataset):
        pass

    # TODO: Handle worker failures (like Lab 1)
    def on_worker_failure(self, worker_id):
        pass
```

**Tasks**:
1. Implement `train_epoch` to distribute batches to workers
2. Aggregate gradients from all workers
3. Handle worker failures and reassign tasks
4. Compare speedup vs. single-worker training

### Exercise 2: Build a Parameter Server (KVRaft Style)

Extend Lab 4 KVRaft to serve as a parameter server:

```python
class ParameterServerKV:
    """
    Use KVRaft as a parameter server

    Key: layer name
    Value: parameters
    """
    def __init__(self, raft_cluster):
        self.kv = KVRaft(raft_cluster)

    # TODO: Implement pull/push operations
    def pull(self, layer_name):
        # Like Get in KVRaft
        pass

    def push(self, layer_name, gradient):
        # Like Put in KVRaft
        # Apply gradient to parameters
        pass

    # TODO: Add versioning (like snapshots)
    def pull_with_version(self, layer_name):
        pass
```

**Tasks**:
1. Implement pull/push using KVRaft operations
2. Add versioning to track parameter updates
3. Benchmark vs. single-machine training
4. Test fault tolerance: kill Raft replicas mid-training

### Exercise 3: Model Sharding (Lab 5 Style)

Implement model parallelism using sharding concepts:

```python
class ModelSharder:
    """
    Partition a large model across GPUs
    Like Lab 5 sharding, but for model layers
    """
    def __init__(self, model, num_shards):
        self.model = model
        self.num_shards = num_shards
        self.shard_config = self.compute_sharding()

    # TODO: Implement sharding strategy
    def compute_sharding(self):
        # Assign layers to GPUs
        pass

    # TODO: Handle forward pass across shards
    def forward(self, input):
        # Pass activations between GPUs
        pass

    # TODO: Implement reconfiguration (Lab 5 style)
    def reconfigure(self, new_num_shards):
        # Migrate layers to new GPUs
        pass
```

**Tasks**:
1. Partition model layers across GPUs
2. Implement forward/backward pass with activation passing
3. Add dynamic reconfiguration when GPUs added/removed
4. Measure memory savings vs. single-GPU

### Exercise 4: Fault-Tolerant Training (Raft Checkpointing)

Add Raft-style fault tolerance to training:

```python
class FaultTolerantTrainer:
    """
    Checkpoint training state like Raft persists state
    """
    def __init__(self, model, checkpoint_dir):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.step = 0

    # TODO: Implement checkpointing (like Raft persist)
    def checkpoint(self):
        pass

    # TODO: Implement recovery (like Raft readPersist)
    def recover(self):
        pass

    # TODO: Add replication for redundancy
    def replicated_train_step(self, batch):
        # Like Raft: compute on multiple replicas
        # Detect divergence
        pass
```

**Tasks**:
1. Checkpoint model/optimizer state periodically
2. Test recovery after simulated crash
3. Add gradient verification using redundant computation
4. Measure checkpoint overhead

---

## Conclusion

The principles taught in MIT 6.5840 are not just theoretical—they are the foundation of modern AI infrastructure:

- **MapReduce** powers data-parallel training of billion-parameter models
- **Raft consensus** enables federated learning across hospitals and phones
- **Sharding** allows us to partition models that don't fit on any single GPU
- **Fault tolerance** keeps $10M training runs alive despite hardware failures
- **Replication** serves billions of inference requests per day
- **Consistency models** determine the speed vs. accuracy trade-offs in distributed training

As AI models grow larger and more complex, distributed systems expertise becomes increasingly critical. The next breakthrough in AI may not come from a new architecture, but from better distributed systems that can train existing architectures more efficiently.

**Key Takeaway**: Understanding distributed systems is no longer optional for AI researchers—it's essential.

---

## References

### Papers Connecting Distributed Systems × AI

1. **MapReduce for ML**: Dean et al., "Large Scale Distributed Deep Networks" (NIPS 2012)
2. **Parameter Servers**: Li et al., "Scaling Distributed Machine Learning with the Parameter Server" (OSDI 2014)
3. **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
4. **Model Parallelism**: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
5. **Fault Tolerance**: Chen et al., "Checkpoint and Restore of Micro-service in Docker Containers" (2015)
6. **Consistency in ML**: Ho et al., "More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server" (NIPS 2013)

### Systems & Frameworks

- **PyTorch Distributed**: https://pytorch.org/tutorials/beginner/dist_overview.html
- **TensorFlow Distributed**: https://www.tensorflow.org/guide/distributed_training
- **Horovod**: https://horovod.readthedocs.io/
- **DeepSpeed**: https://www.deepspeed.ai/
- **Ray**: https://docs.ray.io/en/latest/train/train.html
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM

### Courses

- **MIT 6.5840**: http://nil.csail.mit.edu/6.5840/2025/
- **Stanford CS348K**: Visual Computing Systems
- **CMU 15-418**: Parallel Computer Architecture
- **Berkeley CS294**: AI Systems

---

**Last Updated**: November 2025
**Author**: Generated for MIT 6.5840 Students
**License**: Educational Use Only
