# Quick Reference: Distributed Systems → AI

## One-Page Cheat Sheet

### Lab 1: MapReduce → Data Parallelism

| MapReduce | AI Training |
|-----------|-------------|
| Map tasks | Forward pass on data batch |
| Reduce tasks | Gradient aggregation |
| Coordinator | Parameter server / AllReduce root |
| Worker failure | GPU failure → checkpoint restart |
| Speculative execution | Straggler mitigation |

**Use Cases**: Training large models on massive datasets (GPT, BERT, ImageNet)

### Lab 2: Key-Value Server → Parameter Server

| KV Concept | AI Equivalent |
|------------|---------------|
| Get(key) | Pull(layer_name) → get parameters |
| Put(key, value) | Push(layer_name, gradient) → update params |
| Linearizability | Synchronous SGD |
| Locks | Gradient lock during update |

**Use Cases**: Distributed training with centralized parameter storage

### Lab 3: Raft → Federated Learning

| Raft | Federated Learning |
|------|-------------------|
| Leader election | Aggregator selection |
| Log replication | Model update distribution |
| Commits | Accepted global updates |
| Snapshots | Model checkpoints |
| Persistence | Fault-tolerant training |
| Quorum | Minimum participants for update |

**Use Cases**: Cross-device ML, privacy-preserving training, edge AI

### Lab 4: KVRaft → Replicated Parameter Server

| KVRaft | AI Application |
|--------|----------------|
| State machine replication | Model replica consistency |
| Raft for consensus | Coordinating parameter updates |
| Snapshots | Model versioning |
| Duplicate detection | Idempotent gradient updates |

**Use Cases**: Fault-tolerant training, high-availability serving

### Lab 5: Sharded KV → Model Parallelism

| Sharding | Model Parallelism |
|----------|-------------------|
| Shard = key range | Model partition (layers/tensors) |
| Shard groups | GPU groups for each partition |
| Configuration controller | Orchestrator (Megatron, DeepSpeed) |
| Data migration | Activation passing between GPUs |
| Reconfiguration | Dynamic GPU addition/removal |

**Use Cases**: Training models too large for single GPU (GPT-3, T5, BLOOM)

---

## Consistency Models

```
Strong Sync          Stale Sync        Async
(Slow, Exact)       (Medium)          (Fast, Approx)
     |                   |                  |
     v                   v                  v
Linearizable         Bounded Stale     Eventual
Like Raft           Like SSP          Consistency
```

**When to Use**:
- **Strong**: Small-scale, research, debugging
- **Stale**: Production training, good balance
- **Async**: Very large scale, robust algorithms

---

## Common Patterns

### Pattern 1: Parameter Server (Lab 2 + Lab 3)
```
        PS (Raft replicated)
             ↑    ↓
    ┌────────┼────────┐
Worker 1  Worker 2  Worker 3
```
**Used by**: DistBelief, BytePS, Angel

### Pattern 2: AllReduce Ring (Lab 1 distributed)
```
W1 → W2 → W3 → W4 → W1
```
**Used by**: Horovod, PyTorch DDP, NCCL

### Pattern 3: Hybrid (Lab 1 + Lab 5)
```
Data Parallel Groups
    ↓
┌───┼───┐
G1  G2  G3  (each is model parallel)
```
**Used by**: Megatron-LM, DeepSpeed ZeRO

---

## Fault Tolerance Strategies

| Strategy | Lab Inspiration | AI Implementation |
|----------|----------------|-------------------|
| Checkpointing | Raft persist() | Save model every N steps |
| Replication | Raft replicas | Redundant gradient computation |
| Task reassignment | MapReduce | Restart failed workers |
| Reconfiguration | ShardKV | Elastic training (add/remove GPUs) |
| Log compaction | Raft snapshots | Gradient accumulation |

---

## Quick Decision Tree

**Question: What distributed system concept do I need?**

```
Do you have multiple GPUs?
├─ Yes: Need parallelism
│   ├─ Same data, different batches? → Data Parallel (Lab 1)
│   ├─ Different data shards? → Federated (Lab 3)
│   └─ Model too big for 1 GPU? → Model Parallel (Lab 5)
│
├─ Need fault tolerance?
│   ├─ Checkpoint training → Raft persistence (Lab 3C)
│   └─ High availability serving → Replication (Lab 3)
│
├─ Need to serve models?
│   ├─ Single model, many requests → Replication (Lab 3)
│   └─ Many models → Sharded serving (Lab 5)
│
└─ Need consistency guarantees?
    ├─ Exact gradients → Sync SGD (Linearizable, Lab 2)
    ├─ Bounded staleness → SSP
    └─ Fast, approximate → Async SGD
```

---

## Performance Comparison

### Scaling Efficiency

| Approach | Ideal Speedup | Actual (1000 GPUs) | Efficiency |
|----------|---------------|-------------------|------------|
| Data Parallel (sync) | 1000x | ~700x | 70% |
| Data Parallel (async) | 1000x | ~800x | 80% (lower accuracy) |
| Model Parallel | 1.0x | 0.8x | N/A (memory, not speed) |
| Hybrid (3D) | 1000x | ~650x | 65% (complex) |

### Trade-offs

| Metric | Data ∥ | Model ∥ | Federated | AllReduce |
|--------|--------|---------|-----------|-----------|
| Comm overhead | Medium | High | Very High | Medium |
| Memory saving | No | Yes | Yes | No |
| Scalability | Excellent | Limited | Limited | Good |
| Fault tolerance | Easy | Medium | Hard | Medium |

---

## Common Pitfalls

### From MapReduce (Lab 1)
- **Straggler problem**: Slow GPUs delay training
  - Solution: Dynamic batching, timeout + restart
- **Communication bottleneck**: All workers → coordinator
  - Solution: AllReduce instead of parameter server

### From Raft (Lab 3)
- **Leader bottleneck**: All writes through one node
  - Solution: Sharded parameter servers (multi-leader)
- **Log growth**: Infinite gradient history
  - Solution: Periodic snapshots (like Lab 3D)

### From Sharding (Lab 5)
- **Load imbalance**: Some shards busier than others
  - Solution: Dynamic resharding, work stealing
- **Migration overhead**: Moving model partitions expensive
  - Solution: Pipeline parallelism (minimize movement)

---

## Code Patterns

### Checkpoint (Raft-style)
```python
def checkpoint(self):
    torch.save({
        'step': self.step,
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
    }, f'ckpt_{self.step}.pt')
```

### AllReduce (MapReduce-style)
```python
def all_reduce(gradients):
    # Reduce
    total = sum(gradients)
    # Broadcast
    return [total / len(gradients)] * len(gradients)
```

### Parameter Server (KV-style)
```python
class ParamServer:
    def pull(self, key): return self.params[key]
    def push(self, key, grad): self.params[key] -= lr * grad
```

### Sharding (Lab 5-style)
```python
def shard_model(model, num_gpus):
    layers_per_gpu = len(model.layers) // num_gpus
    for i, gpu in enumerate(gpus):
        start = i * layers_per_gpu
        end = start + layers_per_gpu
        gpu.load(model.layers[start:end])
```

---

## Real Systems Map

| System | Based On | Labs Used |
|--------|----------|-----------|
| Horovod | Ring AllReduce | Lab 1 |
| PyTorch DDP | AllReduce | Lab 1 |
| BytePS | Parameter Server | Lab 2, Lab 3 |
| Megatron-LM | Model Sharding | Lab 5 |
| DeepSpeed ZeRO | Hybrid Sharding | Lab 1 + Lab 5 |
| TF Parameter Server | Parameter Server | Lab 2 |
| Federated Learning | Consensus | Lab 3 |
| Ray Train | MapReduce + Fault Tol | Lab 1 + Lab 3 |

---

## Resources

**Deep Dive**: See `distributed-systems-for-ai-research.md`

**Papers**:
- MapReduce: Dean & Ghemawat, 2004
- Parameter Server: Li et al., OSDI 2014
- Federated Learning: McMahan et al., AISTATS 2017
- Megatron: Shoeybi et al., 2019

**Frameworks**:
- PyTorch Distributed
- DeepSpeed
- Horovod
- Ray

---

**Quick Tip**: Start with data parallelism (Lab 1). Add fault tolerance when needed (Lab 3). Use model parallelism only when model doesn't fit on single device (Lab 5).
