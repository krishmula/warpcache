# warpcache

## Glossary

**Batch size**:
The number of requests that one decode call handles.
_Avoid_: batch (for the count)

**Request**:
One item in a decode call. Each request has its own query, its own history and its own row of the block table.
_Avoid_: batch (for one item), sequence

**History**:
The tokens that a request already has in the KV cache. The history length is how many there are.
_Avoid_: context length, sequence length

**Chunk**:
One back-to-back range of a request's history that one block handles in Path B.
_Avoid_: split, token range

**Workspace**:
The FP32 buffer that holds each chunk's (m, l, u) until the merge.
_Avoid_: scratch buffer

**Partial state**:
A chunk's (m, l, u) for one query head, before the merge. An empty chunk's partial state is m = -inf, l = 0, u = 0.
_Avoid_: running state, chunk state

**Path A**:
The kernel mode with 1 chunk. Each block covers the whole history and writes the normalized output.
_Avoid_: direct decode

**Path B**:
The kernel mode with more than 1 chunk. Each block writes a partial state to the workspace, and PyTorch merges them.
_Avoid_: split decode

**Block**:
One unit that the grid launches, one per grid cell. In this kernel a block is one warp of 32 threads, and "warp" is used only when the 32 threads working together matters, as in the shuffles.
_Avoid_: CTA

**Slot**:
One of the 16 places in a page. A slot holds one token's K and V for every KV head, for one layer. Its number across the whole cache is page × 16 + page slot.

**Page slot**:
A slot's position inside its page, from 0 to 15.
_Avoid_: offset
