struct EmbeddingParams {
    uint num_indices;
    uint embedding_dim;
    uint num_embeddings;
    int padding_idx;
};

[[vk::push_constant]] EmbeddingParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> indicesBuf;
[[vk::binding(1, 0)]] StructuredBuffer<float> weightOrGradOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outOrGradWeight;

[shader("compute")] [numthreads(256, 1, 1)]
void embedding_forward(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = pc.num_indices * pc.embedding_dim;
    if (idx >= total) return;

    uint row = idx / pc.embedding_dim;
    uint col = idx % pc.embedding_dim;
    int tok = int(indicesBuf[row]);

    if (tok == pc.padding_idx || tok < 0 || uint(tok) >= pc.num_embeddings) {
        outOrGradWeight[idx] = 0.0;
    } else {
        outOrGradWeight[idx] = weightOrGradOut[uint(tok) * pc.embedding_dim + col];
    }
}

// Race-free column-parallel backward: each thread owns one (tok, col) element
// or walks the indices for its column `col`, accumulating into `grad_weight`.
// Workgroup grid: x = ceil(embedding_dim / 32), y = ceil(num_embeddings / 8).
[shader("compute")] [numthreads(32, 8, 1)]
void embedding_backward(uint3 tid : SV_DispatchThreadID) {
    uint col = tid.x;
    uint tok = tid.y;
    if (col >= pc.embedding_dim || tok >= pc.num_embeddings) return;

    if (int(tok) == pc.padding_idx) {
        outOrGradWeight[tok * pc.embedding_dim + col] = 0.0;
        return;
    }

    float sum = 0.0;
    for (uint i = 0; i < pc.num_indices; ++i) {
        int idx_tok = int(indicesBuf[i]);
        if (idx_tok == int(tok)) {
            sum += weightOrGradOut[i * pc.embedding_dim + col];
        }
    }
    outOrGradWeight[tok * pc.embedding_dim + col] = sum;
}
