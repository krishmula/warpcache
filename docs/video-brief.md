# Video brief: warpcache, explained from zero

This file is a brief for an AI that writes a 3Blue1Brown-style explainer video
(Manim scenes plus narration). It is self-contained. Every fact, number, and
worked example below comes from the warpcache repository or is computed from
it, and the arithmetic has been checked.

## Part 0. Instructions for the video author

**Audience.** One viewer, the person who built this project. They know Python
and PyTorch. They are new to GPUs and CUDA. They wrote the kernel, but they want
the whole picture to finally click: why each piece exists and how the pieces
fit.

**Goal.** By the end, the viewer can explain these five things out loud:

1. Why generating one token of text is limited by memory, not by math.
2. What a paged KV cache is, and how a kernel finds token 37 of a request.
3. How online softmax computes attention in one pass, and why partial results
   can be merged.
4. How 32 GPU threads (a warp) share one dot product with shuffles.
5. Why INT8 storage and splitting the history into chunks (Path B) each attack
   a different bottleneck, and why a selector picks between them.

**Style.**

- Build each idea from a tiny concrete example before the general rule. Use
  small numbers the viewer can check in their head.
- Name a concept on screen the moment it first appears, then keep that exact
  name. The glossary in Part 6 lists the names.
- Show the "aha" as a visual transformation, not a sentence. For example, the
  merge of two partial softmax states should look like two piles of weight
  being rescaled onto one common baseline.
- Pose a question before you answer it. Let the viewer feel the problem first.
- Keep each scene short (30 to 90 seconds) and give it one idea.
- Use one consistent color code across the video:
  - Q (the query) in yellow.
  - K (keys) in blue.
  - V (values) in green.
  - The running state m, l, u in orange.
  - Memory traffic (bytes moving) in teal.
  - Idle hardware in dark grey.
- Label predictions as predictions. The project has not measured its final
  timings yet (see Part 5).

**Existing assets.** The repo already has five Manim scenes in
`docs/slides/anim/scenes.py`: `ThreadsAndWarps`, `WarpShuffle`, `GridBlocksSMs`,
`OnlineSoftmax`, and `LatencyHiding`. They use a serif font for prose, a mono
font for code and numbers, yellow highlights, and teal for labels. Match that
look so new scenes can sit next to the old ones.

**Length.** Aim for 20 to 30 minutes, in the seven acts below. Each act can be
its own chapter.

---

## Part 1. The project in one paragraph

warpcache is a CS 259 class project. Its title is "Workload Aware CUDA Decode
Attention over an INT8 Paged KV Cache". It is one custom CUDA kernel that runs
the attention step of a language model while the model generates text one token
at a time (the decode phase). The key and value history (the KV cache) is
stored in fixed-size pages, and each number is stored as an 8-bit integer
(INT8) instead of a 16-bit float (FP16), which halves the bytes. The kernel has
two ways to split the work across the GPU. Path A gives each query head one
group of threads that walks the whole history. Path B cuts the history into
chunks, gives each chunk its own group of threads, and then merges the partial
answers. A hand-set selector picks Path A or Path B from the batch size and
context length. The research question is how INT8 storage and the work split
interact across context lengths and batch sizes, and whether a simple selector
beats one fixed choice everywhere.

---

## Part 2. The story, act by act

### Act 1. One token at a time (the decode problem)

**Question to open with.** When a chatbot writes its answer, what does the GPU
actually do for each new word?

**Idea.** A language model writes text one token at a time. For each new token,
every layer runs attention. Attention lets the new token look back at every
token before it.

Build attention for one query head with a tiny example:

- The new token has a query vector **q** (yellow).
- Every earlier token t has a key vector **k_t** (blue) and a value vector
  **v_t** (green).
- Score of token t: z_t = (q · k_t) / sqrt(D), where D = 128 is the head
  dimension.
- Weights: softmax of the scores, p_t = e^(z_t) / Σ e^(z_j).
- Output: Σ p_t · v_t. It is a weighted average of the value vectors.

**Visual.** A row of token boxes. Each box holds a blue K arrow and a green V
arrow. A yellow q arrow sweeps across them, and each box lights up with
brightness equal to its weight. The output is a green arrow built as a blend of
the V arrows.

**Key observation.** The keys and values of old tokens never change. So the
model saves them instead of recomputing them. That saved history is the **KV
cache**. On every decode step, attention must read the entire KV cache once.

The model in this project is Qwen3-1.7B: 28 layers, 16 query heads, 8 KV heads,
head dimension 128. It has fewer KV heads than query heads. This is grouped
query attention (**GQA**). Two query heads share one KV head. Query head h reads
KV head h / 2 (integer division), so query head 7 reads KV head 3.

### Act 2. The bottleneck is memory, not math

**Question.** The GPU does trillions of operations per second. Why is decode
slow?

Count bytes and count math for one layer at a context of 16,384 tokens:

- Bytes per token per layer (FP16): K and V, 8 KV heads, 128 numbers each, 2
  bytes each. 2 × 8 × 128 × 2 = **4,096 bytes = 4 KiB**.
- Whole history for one layer: 16,384 × 4 KiB = **64 MiB**.
- The RTX 3090 used in this project moves about 936 GB/s from its main memory
  (DRAM). Reading 64 MiB takes about **72 microseconds**.
- Math per token per query head: about 256 operations for q · k and 256 for the
  weighted sum of V, so about 512. Over 16 query heads and 16,384 tokens, that
  is about **134 million operations**.
- The 3090 does about 35 trillion FP32 operations per second. 134 million
  operations take about **4 microseconds**.

So the GPU spends about 72 µs waiting for bytes and about 4 µs computing. The
kernel is **memory bound** by a factor of almost 20.

**Visual.** Two progress bars racing. The "math" bar fills almost instantly and
then waits. The "bytes from memory" bar crawls. Then show the ratio as
**arithmetic intensity**: about 2 operations per byte for decode, versus the
3090's balance point of about 38 operations per byte. Draw a simple roofline: a
sloped line (memory limit) that meets a flat line (compute limit) at 38. Put a
dot for decode at x = 2, far down the slope.

**Consequence.** Every design choice in the project follows from this fact.

- Fewer bytes means less time. This motivates INT8 (Act 4).
- To pull bytes at full speed, the GPU must have enough work in flight. This
  motivates Path B (Act 6).

**Evidence from the project.** The first GPU test (Gate 0, October 24, 2026)
ran a tiny "multiply every number by 2" kernel. Nsight Compute, NVIDIA's
profiler, measured it at 87% of peak DRAM throughput and 18% of peak compute.
That is a memory-bound kernel, and decode attention behaves the same way.

### Act 3. The paged KV cache

**Question.** Many users chat at once, and each conversation grows by one token
per step. How do you store all those growing histories without wasting memory?

**Idea.** Borrow the idea of virtual memory from operating systems. Cut GPU
memory into fixed-size **pages** of 16 tokens each. A request owns a list of
pages that do not need to be next to each other. That list is the **block
table**. It maps a logical page number (0, 1, 2, ...) to a physical page number.

**Tensor shapes** (from `docs/scope.md`):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| K cache, V cache | [P, 16, 8, 128] | P physical pages, 16 slots per page, 8 KV heads, 128 numbers |
| K scale, V scale | [P, 16, 8] | One fp32 scale per (token, KV head), INT8 only |
| Block table | [B, max_pages] | For each of B requests, logical page to physical page |
| Sequence lengths | [B] | How many tokens each request has |
| Q | [B, 16, 128] | One query per request per query head |
| Output | [B, 16, 128] | One output vector per request per query head |

**Worked example (put this on screen).** A request has the block table
[9, 2, 5]. Where is its token 37?

1. Logical page = 37 / 16 = 2 (integer division).
2. Offset in the page = 37 mod 16 = 5.
3. Physical page = block_table[2] = 5.
4. Answer: physical page 5, slot 5.

**Visual.** A long strip of physical pages in GPU memory, colored by which
request owns them, scattered and interleaved. One request's block table is a
short list of arrows pointing into the strip. Animate "token 37" as a pointer
that hops through the three steps above.

New tokens get written with a **slot** number: slot = page × 16 + offset. In
this project, writing into the cache happens in PyTorch, not in CUDA.

### Act 4. INT8: storing each number in one byte

**Question.** If time is bytes, can we store fewer bytes?

**Idea.** Symmetric INT8 quantization. For each vector of 128 numbers (one
token, one KV head), find the largest absolute value. Divide it by 127 to get
the **scale**. Store each number as round(x / scale), an integer from −127 to
127. To read it back, multiply by the scale.

**Worked example (4 numbers instead of 128).**

- x = [0.50, −1.30, 0.01, 2.54]
- Largest absolute value = 2.54, so scale = 2.54 / 127 = 0.02.
- x / scale = [25, −65, 0.5, 127]
- Round half to even: [25, −65, 0, 127]
- Read back: [0.50, −1.30, 0.00, 2.54]

The small value 0.01 became 0. The largest number set the scale, and that scale
was too coarse for the small number. This is the whole quantization-error story
in one example.

**Bytes.** Per token per layer, INT8 stores 2 × 8 × 128 × 1 = 2,048 bytes of K
and V, plus 2 × 8 × 4 = 64 bytes of fp32 scales. That is 2,112 bytes, about
**52% of FP16's 4,096 bytes**. If the kernel is truly memory bound and
converting back to floats is cheap, INT8 can nearly halve the time.

**Rules the project fixed** (`reference/quant.py`):

- Rounding is round half to even, which matches CUDA's `__float2int_rn`.
- A vector of all zeros gets scale 1.0, so it reads back as exact zeros.
- Infinity or NaN input is rejected.

**What the measurement found** (CPU, October 11, `scripts/kv_quant_error.py`).
The script ran real text through Qwen3-1.7B, captured the real K and V, and
compared attention output from an INT8 cache against the full-precision cache.

| Model | INT8 error, mean / worst | With K smoothing, mean / worst |
| --- | --- | --- |
| Qwen2.5-1.5B | 2.7% / 40% | 1.5% / 10% |
| Qwen3-1.7B | 6.1% / 33% | 1.8% / 8% |

- V quantizes well, at about 1% error. Almost all error comes from K.
- K has **outlier channels**. In early layers, the largest channel is up to 66
  times the median channel. The outlier sets the scale, so every other channel
  loses precision. This is the 0.01-becomes-0 example at real scale.
- **K smoothing** fixes most of it. Divide each K channel by a fixed factor s
  before storing it, and multiply q by the same s. The dot product q · k does
  not change, but the outliers shrink. This idea is from QServe's
  SmoothAttention (Lin et al., 2024). The class scope cuts it, and the paper
  names it as future work.

**Visual.** A bar chart of the 128 channels of one K vector. One bar towers over
the rest. Draw the 255 quantization levels as horizontal grid lines spaced by
the scale. The tall bar spans them all. The short bars fall between two lines
and snap. Then apply smoothing. The tall bar shrinks, the grid gets finer, and
the short bars stop snapping.

### Act 5. Online softmax: attention in one pass

**Question.** Softmax needs the sum of all e^(z) before it can weight anything.
Must the kernel read the whole history twice, once to sum and once to weight?

**Idea.** No. Keep a running state of three things and update it after each
token:

- **m**, the largest score seen so far.
- **l**, the sum of e^(z − m) over tokens seen so far.
- **u**, the sum of e^(z − m) · v over tokens seen so far. It is a vector of
  128 numbers.

When a new token with score z arrives:

```
new_m  = max(m, z)
shrink = e^(m − new_m)       # rescale the old sums to the new baseline
w      = e^(z − new_m)       # weight of the new token
l      = l · shrink + w
u      = u · shrink + w · v
m      = new_m
```

At the end, the answer is u / l.

Subtracting m keeps every exponent at most 0, so e^(...) never overflows. The
"shrink" factor is the clever part. When a bigger score shows up, every old
term was measured against a baseline that is now too low, so each old term is
multiplied by e^(old m − new m) to move it to the new baseline.

**Worked example (scalar V so it fits on screen).** Scores z = [1, 3, 2],
values v = [10, 20, 30].

| Step | z | new m | shrink | w | l | u |
| --- | --- | --- | --- | --- | --- | --- |
| start | | −∞ | | | 0 | 0 |
| 1 | 1 | 1 | 0 | 1 | 1 | 10 |
| 2 | 3 | 3 | e^−2 = 0.135 | 1 | 1.135 | 21.35 |
| 3 | 2 | 3 | 1 | e^−1 = 0.368 | 1.503 | 32.39 |

Answer: u / l = 32.39 / 1.503 = **21.55**.

Check with ordinary softmax: e^1, e^3, e^2 = 2.718, 20.09, 7.389. Their sum is
30.19. The weights are 0.090, 0.665, 0.245. The output is 0.090 × 10 + 0.665 ×
20 + 0.245 × 30 = **21.55**. Same answer.

**Visual.** Three stacks of blocks, one per token, with heights e^(z − m). When
token 2 arrives with a bigger score, the baseline jumps up and the earlier stack
visibly shrinks by e^−2. The u total is drawn as a green bar that shrinks and
grows in step.

**Merging partial states.** The same rescaling lets you merge two partial
states computed on different parts of the history. Given states (m₁, l₁, u₁)
and (m₂, l₂, u₂):

```
m = max(m₁, m₂)
l = l₁ · e^(m₁ − m) + l₂ · e^(m₂ − m)
u = u₁ · e^(m₁ − m) + u₂ · e^(m₂ − m)
answer = u / l
```

**Merge example.** Split the same three tokens into chunk 1 = tokens {1, 2}
(scores 1 and 3) and chunk 2 = token {3} (score 2).

- Chunk 1 alone gives m = 3, l = 1.135, u = 21.35. These are the values after
  step 2 above.
- Chunk 2 alone gives m = 2, l = 1, u = 30.
- Merge: m = 3. Chunk 1 weight e^0 = 1, chunk 2 weight e^−1 = 0.368.
- l = 1.135 + 0.368 = 1.503, u = 21.35 + 0.368 × 30 = 32.39.
- Answer 21.55. Same as one pass.

An empty chunk has m = −∞, l = 0, u = 0. Its weight is e^(−∞) = 0, so it adds
nothing. This lets the kernel use more chunks than there are tokens.

**Why this matters.** The merge is the mathematical license for Path B. Any
split of the history into chunks gives the exact same answer, up to rounding.
This is the idea behind Flash-Decoding, and behind vLLM's paged attention v2.
It comes from Milakov and Gimelshein (2018), "Online normalizer calculation
for softmax".

The repo implements this three times so each can check the others:

- `reference/attention.py` computes it directly with PyTorch softmax, and also
  splits it into chunks with `partial_state` and `merge_states`.
- `reference/online_softmax.py` runs the token loop above, shaped like one GPU
  thread group.
- `kernels/decode.cu` runs it on the GPU.

### Act 6. How a GPU runs the kernel

Build the GPU vocabulary bottom up. The existing scenes `ThreadsAndWarps`,
`GridBlocksSMs`, and `WarpShuffle` cover parts of this.

**Thread, warp, block, grid, SM.**

- A **thread** runs the kernel code with its own registers.
- A **warp** is 32 threads that execute each instruction together, in lockstep.
  Each thread in a warp is a **lane**, numbered 0 to 31.
- A **block** is a group of threads that runs on one streaming multiprocessor.
  In this kernel, a block is exactly one warp.
- A **grid** is all the blocks of one kernel launch.
- A **streaming multiprocessor (SM)** is one of the GPU's cores. The RTX 3090
  has **82 SMs**.

**How the decode kernel maps work** (`kernels/decode.cu`, version 1):

- Grid = (16 query heads, B requests). One block per (request, query head).
- Each block is one warp of 32 lanes.
- A head vector has 128 numbers, so lane i owns numbers 4i to 4i + 3. Lane 0
  owns 0 to 3. Lane 31 owns 124 to 127.

**The kernel, step by step, for one warp:**

1. **Who am I.** Read the request b, query head h, lane, KV head = h / 2, and
   the history length.
2. **Load my piece of q.** Each lane loads its 4 numbers of q and converts them
   from fp16 to fp32.
3. **Start the state.** m = −∞, l = 0, and 4 numbers of u = 0.
4. **For each token t in the history:**
   - Find the token through the block table: physical page =
     block_table[t / 16], slot = t mod 16. Then load this lane's 4 numbers of
     K and 4 of V.
   - Each lane multiplies its 4 q numbers with its 4 K numbers and adds them.
     This is a **partial dot product**.
   - Combine the 32 partial sums into the full score with **warp shuffles**
     (below). Divide by sqrt(128).
   - Run the online softmax update. Every lane has the same score, so every
     lane computes the same m and l, and each lane updates its own 4 numbers of
     u.
5. **Write the output.** Each lane writes its 4 numbers of u / l as fp16.

**Warp shuffle, the butterfly.** `__shfl_xor_sync(mask, value, d)` lets lane i
read a register value from lane (i XOR d), with no trip through memory. Five
rounds with d = 16, 8, 4, 2, 1 give every lane the sum of all 32:

- Round 1 (d = 16): lane 0 pairs with lane 16, lane 5 with 21. Each lane adds
  its partner's value. Now each lane holds a sum of 2.
- Round 2 (d = 8): lane 0 pairs with 8. Each lane now holds a sum of 4.
- Rounds 3, 4, 5: sums of 8, 16, then 32.

After 5 rounds (log₂ 32 = 5), all 32 lanes hold the full dot product.

**Visual.** 32 lanes in a row, each holding a number. Draw crossing arcs
between partners each round, like the butterfly diagram in a fast Fourier
transform. After each round, every lane's number grows. After round 5, every
lane shows the same total.

The project practiced this first on a warm-up kernel, `kernels/warp_reduce.cu`.
One warp computes the sum and max of a row of 128 floats with the same five
shuffle rounds.

**Latency hiding.** Loading a token's K and V from DRAM takes hundreds of clock
cycles. While one warp waits, the SM switches to another warp that is ready to
compute. The GPU stays fast only if each SM has many warps to switch among. The
existing scene `LatencyHiding` shows this as timelines of warps with waiting
gaps that interleave on a shared memory bus.

### Act 7. Path A, Path B, and the selector

**The problem with Path A.** Path A launches one block per (request, query
head). At batch 1, that is 16 blocks on a GPU with 82 SMs.

- 66 of the 82 SMs have no work at all.
- The 16 busy SMs each run one warp, so there is nothing to switch to while a
  load is in flight.
- Each warp walks the entire history, one token after another. At 16K context,
  that is 16,384 steps in a row.

The memory system can deliver far more than 16 lonely warps can ask for. The
kernel is no longer limited by memory bandwidth. It is limited by **latency**,
the time for each step to wait on its load.

**Visual.** The 82 SMs as a grid of squares. At batch 1, 16 light up and 66 stay
dark grey. Each lit SM shows one warp walking a very long line of tokens.

**Path B: split the history.** Cut each history into C chunks. Launch one block
per (request, query head, chunk). Each block runs the same online softmax over
its token range and writes a partial state (m, l, u) to a scratch buffer, the
**workspace**. PyTorch then merges the C states per head with the merge rule
from Act 5.

- At batch 1 with 64 chunks: 16 × 64 = 1,024 warps instead of 16.
- Each warp walks 256 tokens instead of 16,384.
- Cost: one extra step to merge, and a little extra memory traffic for the
  workspace.

**Visual.** The same long line of tokens is cut into 64 short segments. The dark
SMs light up. The segments run side by side, then funnel into a merge node.

**One kernel, both paths.** The design uses one kernel over a token range. With
1 chunk, it normalizes and writes the output (Path A). With more than 1 chunk,
it writes (m, l, u) to the workspace (Path B). The kernel is also a C++
template on the cache element type, so the same source serves FP16 and INT8.

**The factorial question.** Two choices, two values each:

| | Path A | Path B |
| --- | --- | --- |
| FP16 cache | baseline kernel | more warps |
| INT8 cache | fewer bytes | more warps and fewer bytes |

INT8 helps only when bytes are the bottleneck. Path B helps only when there are
too few warps to fill the chip. At small batch and long context, the kernel is
latency bound, so Path B matters most and INT8 matters little. At large batch,
the chip is already full, the kernel becomes memory bound, and INT8 matters
more. The project's question is where these regimes cross over.

**The selector.** A hand-set rule, written from the measurements, picks Path A
or Path B before launch. A candidate rule is "split when batch × query heads is
below the SM count". The paper labels it as hand-set, not learned.

**Predicted numbers (a model, not measurements).**
`scripts/expected_results.py` predicts latency with a simple model: the slower
of the memory time and the per-token latency time, plus launch overhead, plus
25 µs for the PyTorch merge. Its per-token costs are guesses. Show these as
"what we expect", with a clear "prediction" label:

| Batch | Context | SDPA FP16 | Path A FP16 | Path A INT8 | Path B FP16 | Path B INT8 | Best chunks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,024 | 16 µs | 310 µs | 341 µs | 33 µs | 33 µs | 64 |
| 1 | 16,384 | 106 µs | 4,918 µs | 5,410 µs | 112 µs | 112 µs | 64 |
| 16 | 16,384 | 1,540 µs | 4,918 µs | 5,410 µs | 1,378 µs | 1,084 µs | 4 |

The story these predictions tell: Path A version 1 is far slower than the
PyTorch baseline everywhere, because one warp walking tokens one by one is
latency bound. Path B closes the gap by adding warps. At batch 16 and 16K
context, Path B with INT8 is predicted to beat the baseline, because the chip
is full and INT8's smaller bytes finally pay off.

The baseline is PyTorch's built-in attention (SDPA) on an FP16 history that has
been gathered into one contiguous tensor.

---

## Part 3. How the project proves correctness

Use this as a short closing scene about trust. Each layer is checked against
the one before it.

1. **Direct reference.** `reference/attention.py`, `decode_attention`. This is
   plain PyTorch softmax, and it is the ground truth.
2. **Split reference.** `split_decode_attention` must match the direct form for
   any number of chunks, including more chunks than tokens.
3. **Paged reference.** `reference/paged.py` stores tokens in a paged cache,
   optionally INT8, gathers them back through the block table, and runs the
   reference. It is the correctness oracle for the kernel.
4. **Token loop.** `reference/online_softmax.py` must match the reference at
   lengths 1, 7, and 300.
5. **Warp warm-up.** `kernels/warp_reduce.cu` must match `torch.sum` and
   `torch.amax`.
6. **The kernel.** `tests/test_decode.py` checks the CUDA kernel against the
   paged reference:
   - One token in history returns exactly that token's V row.
   - Lengths 2, 15, 16, 17, 300, and 2,048. The page boundary at 16 is where
     indexing bugs hide.
   - A batch of lengths [1, 15, 16, 17, 50] with shuffled pages.
   - Query scaled by 30, so scores are huge, and the output must stay finite.
     This tests the subtract-the-max trick.
   - Bad inputs are rejected, for example an int64 block table or an INT8
     cache without scales.
7. **Memory safety.** `compute-sanitizer` reports 0 out-of-bounds reads.

---

## Part 4. Hardware and setup facts

- **GPU.** One NVIDIA RTX 3090: 24 GB memory, 82 SMs, 6 MB L2 cache, about
  936 GB/s DRAM bandwidth. It runs on a Vast.ai Ubuntu 22.04 virtual machine.
- **Why that host.** The profiler (Nsight Compute) needs access to hardware
  counters. RunPod RTX 4090 containers blocked it with `ERR_NVGPUCTRPERM`.
  Lambda said its VMs do not support Nsight Compute. The Vast.ai VM gives root,
  so counters work.
- **Toolchain.** Driver 580.95.05, CUDA 13.0, torch 2.14.1+cu130, Python 3.12.
- **Workflow.** Code is written on a Mac, which cannot run CUDA. The GPU VM is
  used only to compile, test, and measure, because it bills by the hour.
- **How the kernel reaches Python.** `torch.utils.cpp_extension.load_inline`
  compiles `decode.cu` on first call. The C++ wrapper checks every tensor's
  shape, type, device, and layout with `TORCH_CHECK`, then launches the kernel
  with `<<<grid, 32>>>`.
- **Timing caution.** The L2 cache is 6 MB, so benchmark data must be larger
  than 12 MiB. Otherwise repeated runs read from a warm cache and look faster
  than they really are.

---

## Part 5. Where the project stands (October 31, 2026)

Show this honestly near the end. The video should not claim results that do not
exist yet.

**Done:**

- Scope and tensor contract.
- PyTorch reference for quantization, paged cache, direct and split decode,
  and merge, with tests passing on CPU.
- INT8 error measured on real K and V from Qwen3-1.7B.
- Gate 0: the GPU host works, including profiler counters.
- Online softmax token loop in Python.
- Warp reduction warm-up kernel.
- **Path A FP16 decode kernel, version 1.** It passes 10 tests on the 3090, and
  `compute-sanitizer` reports 0 errors.
- INT8 host path. The C++ wrapper validates INT8 inputs and the INT8 tests
  exist.

**In progress:** INT8 inside the kernel. The wrapper already launches
`decode_kernel<int8_t>` with scale pointers, but the kernel itself is not yet a
template and does not take scales. So the file does not compile at this
commit. The next step is to template the kernel on the cache type and multiply
each loaded INT8 value by its token's scale.

**Not started:** Path B in the kernel, timing against SDPA, Nsight profiles, the
selector, figures, and the paper.

**Cut from the class scope:** a CUDA kernel for cache writes, FlashInfer as a
baseline, K smoothing, sharing one KV head load across its two query heads,
mixed-length batches, a CUDA merge kernel (unless time allows), and vLLM
integration.

**Schedule:** class completion Thursday, November 5, 2026. Submission Monday,
November 9.

---

## Part 6. Glossary for on-screen labels

Use these exact names in the video.

| Term | Meaning |
| --- | --- |
| Decode | Generating one new token, given all earlier tokens. |
| Query (q) | The new token's vector that asks "which earlier tokens matter?" |
| Key (k) | An earlier token's vector that q is compared against. |
| Value (v) | An earlier token's vector that gets averaged into the output. |
| KV cache | The saved keys and values of all earlier tokens. |
| Head dimension (D) | Length of one head's vector. 128 here. |
| GQA | Grouped query attention. Several query heads share one KV head. Here 16 query heads share 8 KV heads. |
| Page | A fixed block of 16 token slots in the cache. |
| Block table | A request's list mapping logical pages to physical pages. |
| Slot | A token's position in the cache, page × 16 + offset. |
| INT8 | Storing each number as an 8-bit integer from −127 to 127. |
| Scale | The fp32 multiplier that turns INT8 back into a real number. One per (token, KV head). |
| Outlier channel | A position in K whose values are much larger than the others. |
| K smoothing | Dividing K channels by fixed factors and multiplying q by the same factors. Cut from scope. |
| Online softmax | Softmax computed in one pass with a running state. |
| State (m, l, u) | Running max score, running weight sum, running weighted sum of V. |
| Merge | Combining partial states from different chunks into one answer. |
| Memory bound | Limited by how fast bytes arrive from memory. |
| Latency bound | Limited by how long each dependent step waits, because too little work is in flight. |
| Arithmetic intensity | Operations per byte read. |
| Thread, lane | One execution unit. A lane is a thread's index within its warp. |
| Warp | 32 threads that run each instruction together. |
| Block | A group of threads on one SM. One warp in this kernel. |
| Grid | All blocks of one launch. |
| SM | Streaming multiprocessor. One of the GPU's 82 cores. |
| Warp shuffle | A lane reads another lane's register directly. |
| Path A | One block per (request, query head) walks the whole history. |
| Path B | One block per (request, query head, chunk), then a merge. |
| Workspace | Scratch memory where Path B writes partial states. |
| Selector | A hand-set rule that picks Path A or Path B before launch. |
| SDPA | PyTorch's built-in attention. The baseline. |

---

## Part 7. Suggested scene list

| # | Scene | One idea | Reuse |
| --- | --- | --- | --- |
| 1 | `TokenByToken` | A model writes one token at a time, and each token looks back at all others. | |
| 2 | `AttentionAsAverage` | Attention is a softmax-weighted average of V. | |
| 3 | `WhyCacheKV` | Old K and V never change, so they are cached and read every step. | |
| 4 | `BytesVsMath` | 72 µs of bytes versus 4 µs of math. Roofline dot at 2 ops per byte. | |
| 5 | `PagedCache` | Pages, block table, and the token 37 lookup. | |
| 6 | `Int8Quantize` | The 4-number quantization example, with 0.01 snapping to 0. | |
| 7 | `KOutliers` | One tall channel ruins the scale. Smoothing shrinks it. | |
| 8 | `OnlineSoftmaxStep` | The 3-token table, with stacks shrinking when the baseline rises. | `OnlineSoftmax` |
| 9 | `MergeStates` | Two chunks merge to the same 21.55. | `OnlineSoftmax` |
| 10 | `ThreadsWarpsBlocks` | Thread, warp, block, grid, SM. Lane i owns 4 numbers. | `ThreadsAndWarps`, `GridBlocksSMs` |
| 11 | `ButterflyShuffle` | Five xor rounds give every lane the full dot product. | `WarpShuffle` |
| 12 | `KernelWalkthrough` | One warp's loop, step by step, next to the CUDA code. | |
| 13 | `LatencyHiding` | Waiting warps need neighbors to switch to. | `LatencyHiding` |
| 14 | `IdleChip` | Path A at batch 1 lights 16 of 82 SMs. | `GridBlocksSMs` |
| 15 | `SplitTheHistory` | Path B cuts the history and fills the chip, then merges. | |
| 16 | `TwoByTwo` | INT8 versus FP16 times Path A versus Path B, and where each wins. | |
| 17 | `TrustLadder` | Each layer checked against the layer below. | |
| 18 | `WhereWeAre` | Done, in progress, next. | |
