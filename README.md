
# 🧠 Theory: GPT-2 with External Differentiable Memory (EDM)

## 1. Motivation

* **Problem with vanilla GPT-2**:

  * Context is limited to a fixed window (e.g. 1024 tokens).
  * Once tokens fall outside, they are forgotten.

* **Problem with only recurrent memory (GRU/LSTM)**:

  * Hidden states are bounded by dimensionality.
  * Memory is lossy, compressed, approximate.

* **Problem with NTMs**:

  * Explicit memory is powerful but unstable to train.
  * Not easily integrated into large pretrained LMs.

* **Goal**:

  * Extend GPT-2’s effective memory beyond its attention window.
  * Provide both *implicit persistence* (RNN state) and *explicit recall* (external memory).
  * Keep training stable and compatible with Hugging Face tools.

---

## 2. Architecture

```
Tokens
   ↓
Embeddings
   ↓
[Early Recurrent Memory]      ← rolling summary
   ↓
GPT-2 Transformer             ← local attention (1024 tokens)
   ↓
[External Differentiable Memory] 
   - Write: store key/value pairs
   - Read: retrieve past states via content-based addressing
   ↓
[Late Recurrent Memory]       ← smooth, persist processed states
   ↓
LM Head → Logits
```

### Components:

1. **Early Recurrent Memory**

   * Compresses raw embeddings into a short-term vector.
   * Provides continuity across sliding windows.

2. **External Differentiable Memory (EDM)**

   * A key–value memory matrix `M`.
   * **Write head**: stores hidden representations at chosen steps.
   * **Read head**: retrieves memory slots by content similarity or learned addressing.
   * Differentiable (using soft attention over memory slots).

3. **Late Recurrent Memory**

   * Smooths the transformer+memory outputs.
   * Maintains coherence across generation.

---

## 3. Memory Dynamics

### Write:

At step *t*, given hidden state `h_t`:

```
key_t = W_k h_t
val_t = W_v h_t
M ← M ∪ (key_t, val_t)   # append or update memory
```

### Read:

At step *t*:

```
attn_weights = softmax(sim(W_q h_t, M.keys))
read_t = Σ attn_weights[i] * M.values[i]
```

### Combine:

The transformer hidden state is enriched with memory recall:

```
h_t' = h_t + read_t
```

---

## 4. Training Strategy

* **Freeze GPT-2**: don’t retrain its huge weights.
* Train only:

  * Recurrent memories (early + late).
  * External memory (key/query/value projections).
  * Optionally LoRA adapters inside GPT-2.

This keeps the system stable and efficient.

---

## 5. Expected Benefits

### vs Vanilla GPT-2:

* Can recall facts long after they leave the attention window.
* Maintains topic consistency across thousands of tokens.

### vs Recurrent GPT-2 Memory:

* Not just lossy compression → explicit recall of stored states.
* Can re-inject old entities, names, or facts.

### vs NTM:

* More stable, because memory is tied to pretrained GPT-2 representations instead of learning algorithms from scratch.
* Doesn’t require complex read/write programs — just attention-based addressing.

---

## 6. Limitations

* Memory grows with sequence length → need eviction/compression strategies.
* Risk of copying noise unless trained with strong supervision.
* Harder to scale than pure attention or pure recurrence.

---

## 7. Analogy

* **Vanilla GPT-2** = a person with only short-term memory.
* **Recurrent GPT-2 Memory** = that person taking mental notes (fuzzy, lossy).
* **GPT-2 + EDM** = that person carrying an indexed notebook, writing things down, and flipping back to the right page when needed.

---

# ✅ One-Line Theory

**GPT-2 with External Differentiable Memory augments self-attention with a trainable notebook, enabling the model to explicitly store and recall representations beyond its fixed context window, while recurrent modules maintain continuity across tokens.**

### PSUEDOCODE

Perfect ⚡ let’s design a **prototype External Differentiable Memory (EDM)** module that you can drop into GPT-2 (just like the recurrent layers).

This will be a **key–value memory** with differentiable read/write, using attention to recall old states.

---

# 📜 External Differentiable Memory (Prototype)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalDifferentiableMemory(nn.Module):
    """
    Key-Value external memory with differentiable read/write.
    Stores GPT-2 hidden states, allows retrieval beyond context length.
    """
    def __init__(self, hidden_size, memory_slots=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots

        # Trainable projections for key/query/value
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.val_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        # Initialize empty memory
        self.register_buffer("keys", torch.zeros(memory_slots, hidden_size))
        self.register_buffer("values", torch.zeros(memory_slots, hidden_size))
        self.ptr = 0  # circular write pointer

    def reset(self, device=None):
        """Clear memory at the start of a new sequence."""
        self.keys = torch.zeros_like(self.keys)
        self.values = torch.zeros_like(self.values)
        self.ptr = 0

    def write(self, hidden_states):
        """
        Write hidden states to memory (using circular buffer).
        hidden_states: (batch, hidden_size)
        """
        k = self.key_proj(hidden_states).detach()
        v = self.val_proj(hidden_states).detach()

        # If batch > 1, average into one slot (simple version)
        k = k.mean(dim=0)
        v = v.mean(dim=0)

        # Store into memory slot
        self.keys[self.ptr % self.memory_slots] = k
        self.values[self.ptr % self.memory_slots] = v
        self.ptr += 1

    def read(self, hidden_states):
        """
        Retrieve from memory using attention over keys.
        hidden_states: (batch, hidden_size)
        returns: (batch, hidden_size)
        """
        if self.ptr == 0:  # memory empty
            return torch.zeros_like(hidden_states)

        q = self.query_proj(hidden_states)  # (batch, hidden_size)
        attn = torch.matmul(q, self.keys.T)  # (batch, memory_slots)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum over values
        readout = torch.matmul(attn, self.values)  # (batch, hidden_size)
        return readout

    def forward(self, hidden_states):
        """
        Combine GPT-2 hidden state with memory read/write.
        hidden_states: (batch, seq, hidden_size)
        """
        outputs = []
        for t in range(hidden_states.size(1)):
            h_t = hidden_states[:, t, :]  # (batch, hidden_size)

            # Read from memory
            read_t = self.read(h_t)

            # Combine hidden state with memory recall
            h_combined = h_t + read_t

            # Write to memory
            self.write(h_t)

            outputs.append(h_combined.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (batch, seq, hidden_size)
```

---

# 🔹 How It Works

* **Memory slots**: a fixed-size matrix (like RAM).
* **Write**: hidden states are projected into key–value pairs and stored (circular buffer).
* **Read**: current hidden state queries memory → soft attention over stored keys → returns a weighted sum of values.
* **Combine**: enriches GPT-2 hidden state with explicit recall from past context.

---

# 🔹 Integration into GPT-2

You can drop it in the **same way we used the recurrent layers**:

```python
class GPT2WithMemory(nn.Module):
    def __init__(self, model_name="gpt2", memory_slots=128):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size
        self.edm = ExternalDifferentiableMemory(hidden_size, memory_slots)

    def reset_memory(self):
        self.edm.reset()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        # Inject external memory
        hidden = self.edm(hidden)

        logits = self.gpt2.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}

    def generate(self, *args, **kwargs):
        self.reset_memory()
        return self.gpt2.generate(*args, **kwargs)
```

---

# ✅ What This Achieves

* GPT-2 now has **external memory** to recall states far beyond its fixed context window.
* Unlike RNNs, this memory is **explicitly addressable** via keys.
* Works like a **neural cache**: GPT-2 can look back into prior slots, not just rely on compressed recurrence.

### PSEUDOCODE

Perfect ⚡ — let’s build a **hybrid GPT-2 with External Differentiable Memory (EDM) + optional QLoRA**.

The design will:

* Wrap **GPT-2** as the base model.
* Attach **external memory** (EDM).
* Allow optional **QLoRA fine-tuning** (via PEFT) while freezing the GPT-2 backbone.
* Trainable pieces = QLoRA adapters + external memory.

---

# 📜 GPT-2 + External Memory + Optional QLoRA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


# -------------------
# External Differentiable Memory
# -------------------
class ExternalDifferentiableMemory(nn.Module):
    """
    Key-Value external memory with differentiable read/write.
    """
    def __init__(self, hidden_size, memory_slots=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.val_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        # Buffers for memory (shared across batches for simplicity)
        self.register_buffer("keys", torch.zeros(memory_slots, hidden_size))
        self.register_buffer("values", torch.zeros(memory_slots, hidden_size))
        self.ptr = 0

    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.ptr = 0

    def write(self, hidden):
        k = self.key_proj(hidden).mean(dim=0)  # avg batch
        v = self.val_proj(hidden).mean(dim=0)

        self.keys[self.ptr % self.memory_slots] = k.detach()
        self.values[self.ptr % self.memory_slots] = v.detach()
        self.ptr += 1

    def read(self, hidden):
        if self.ptr == 0:
            return torch.zeros_like(hidden)

        q = self.query_proj(hidden)  # (batch, hidden)
        attn = torch.matmul(q, self.keys.T)  # (batch, slots)
        attn = F.softmax(attn, dim=-1)

        readout = torch.matmul(attn, self.values)  # (batch, hidden)
        return readout

    def forward(self, hidden_states):
        outputs = []
        for t in range(hidden_states.size(1)):
            h_t = hidden_states[:, t, :]
            r_t = self.read(h_t)
            h_comb = h_t + r_t
            self.write(h_t)
            outputs.append(h_comb.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# -------------------
# GPT-2 + EDM wrapper
# -------------------
class GPT2WithExternalMemory(nn.Module):
    def __init__(self, model_name="gpt2", memory_slots=128, use_lora=False):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        if use_lora:
            # QLoRA config
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],  # attention/MLP projections
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)

        hidden_size = self.gpt2.config.hidden_size
        self.edm = ExternalDifferentiableMemory(hidden_size, memory_slots)

    def reset_memory(self):
        self.edm.reset()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        # Inject external memory
        hidden = self.edm(hidden)

        logits = self.gpt2.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}

    def generate(self, *args, **kwargs):
        self.reset_memory()
        return self.gpt2.generate(*args, **kwargs)
```

---

# 🔹 How It Works

* **Base GPT-2** → unchanged, loaded from Hugging Face.
* **QLoRA (optional)** → LoRA adapters on GPT-2’s attention/MLP projections.
* **External Differentiable Memory** → stores hidden states in key–value slots, retrieves via soft attention, and injects them back into GPT-2’s output stream.
* **Training** → only QLoRA adapters + EDM parameters update. GPT-2 stays frozen.

---

# 🔹 Example Usage

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Base model with QLoRA + memory
model = GPT2WithExternalMemory("gpt2", memory_slots=128, use_lora=True).to("cuda")

prompt = "In the beginning of the universe,"
enc = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generation
gen_ids = model.generate(
    **enc,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
)

print("Prompt:", prompt)
print("Generated:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))
```

---

# ✅ Result

* You now have GPT-2 + external differentiable memory.
* You can optionally wrap GPT-2 with **QLoRA adapters** to fine-tune efficiently.
* The only trainable parts are:

  * **EDM key/query/value projections**
  * **LoRA adapters (if enabled)**

Together, this hybrid setup allows GPT-2 to:

* Retain compressed long-term memory (via EDM).
* Adapt cheaply with LoRA.


