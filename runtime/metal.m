#include "metal_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "metal_kernels.h"
#include <fcntl.h>
#include <limits.h>

static const unsigned prompt_tile_tokens[PROMPT_TILE_COUNT] = {32, 64, 24, 40};

void plamo_metal_residency_add(struct plamo_metal *m, id<MTLResource> buffer) {
    if (!buffer)
        return;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency addAllocation:buffer];
        m->residency_dirty = 1;
    }
#else
    (void)m;
    (void)buffer;
#endif
}

void plamo_metal_residency_remove(struct plamo_metal *m, id<MTLResource> buffer) {
    if (!buffer)
        return;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency removeAllocation:buffer];
        m->residency_dirty = 1;
    }
#else
    (void)m;
    (void)buffer;
#endif
}

void plamo_metal_residency_commit(struct plamo_metal *m) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (m->residency_dirty) {
            [m->residency commit];
            m->residency_dirty = 0;
        }
    }
#else
    (void)m;
#endif
}

/* Under memory pressure macOS can compress or swap an idle process's GPU
   allocations even while a residency set holds them; the next request then paid
   seconds of page-ins before its first GPU command. Requesting residency again
   on idle ticks keeps the retained weights hot (cheap while they are resident). */
void plamo_metal_keep_resident(struct plamo_metal *m) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    // Streaming retains no weights; PLAMO_METAL_KEEP_RESIDENT=0 opts out.
    static int disabled = -1;
    if (disabled < 0) {
        const char *setting = getenv("PLAMO_METAL_KEEP_RESIDENT");
        disabled = setting && !strcmp(setting, "0");
    }
    if (@available(macOS 15.0, *)) {
        if (m && m->residency && !disabled && m->config.memory != PLAMO_METAL_STREAM) {
            plamo_metal_residency_commit(m);
            [m->residency requestResidency];
        }
    }
#else
    (void)m;
#endif
}

int plamo_metal_problem(struct plamo_metal *m, const char *message) {
    ++m->info.errors;
    snprintf(m->info.note, sizeof m->info.note, "%s", message);
    return 0;
}

int plamo_metal_ensure_buffer(struct plamo_metal *m, id<MTLBuffer> *buffer, uint64_t bytes) {
    if (*buffer && (*buffer).length >= bytes)
        return 1;
    if (bytes > m->device.maxBufferLength)
        return plamo_metal_problem(m, "Metal activation buffer exceeds the device limit");
    id<MTLBuffer> replacement =
        [m->device newBufferWithLength:(NSUInteger)plamo_metal_aligned(bytes)
                               options:MTLResourceStorageModeShared];
    if (!replacement)
        return plamo_metal_problem(m, "cannot allocate Metal activation buffer");
    int replacing = *buffer != nil;
    if (*buffer)
        plamo_metal_residency_remove(m, *buffer);
    [*buffer release];
    *buffer = replacement;
    plamo_metal_residency_add(m, replacement);
    if (replacing)
        plamo_metal_residency_commit(m);
    return 1;
}

void plamo_metal_encode_linear(struct plamo_metal *m, id<MTLComputeCommandEncoder> encoder,
                               id<MTLResource> w, NSUInteger offset, unsigned packed,
                               id<MTLBuffer> x, uint64_t xs, id<MTLBuffer> y,
                               NSUInteger output_offset, uint64_t ys, uint64_t rows,
                               uint64_t columns, uint64_t tokens) {
    struct {
        uint32_t rows, columns, tokens, xs, ys, aligned_weights, weight_offset, first_fragment;
    } params = {(uint32_t)rows,
                (uint32_t)columns,
                (uint32_t)tokens,
                (uint32_t)xs,
                (uint32_t)ys,
                !(offset % 8),
                packed ? (uint32_t)(offset / 2) : 0,
                0};
    if (m->draft_quantized && packed == 2 && tokens == 1 && !(columns % 32) &&
        !(params.weight_offset % 32) && !(xs % 4)) {
        id<MTLBuffer> half_input = x == m->normalized ? m->draft_normalized
                                   : x == m->gate     ? m->draft_gate
                                                      : nil;
        // The head (107520 rows) is fastest with four lanes per row; layer
        // projections with sixteen.
        unsigned tall = rows >= 65536;
        if (half_input && m->draft_block[tall]) {
            unsigned row_tile = tall ? 32 : 8;
            [encoder setComputePipelineState:m->draft_block[tall]];
            [encoder setBuffer:(id<MTLBuffer>)w offset:0 atIndex:0];
            [encoder setBuffer:half_input offset:0 atIndex:1];
            [encoder setBuffer:y offset:output_offset atIndex:2];
            [encoder setBytes:&params length:sizeof params atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + row_tile - 1) / row_tile, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            return;
        }
        if (half_input) {
            unsigned wide = rows >= 4096 && columns <= 4096, row_tile = wide ? 16 : 4;
            [encoder setComputePipelineState:m->draft_dot[wide]];
            [encoder setBuffer:(id<MTLBuffer>)w offset:0 atIndex:0];
            [encoder setBuffer:half_input offset:0 atIndex:1];
            [encoder setBuffer:y offset:output_offset atIndex:2];
            [encoder setBytes:&params length:sizeof params atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + row_tile - 1) / row_tile, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            return;
        }
    }
    if (packed == 2 && tokens == 1 && rows >= 4096 && columns <= 4096 && !(columns % 32) &&
        !(params.weight_offset % 32) && !(xs % 4)) {
        [encoder setComputePipelineState:m->q4_rows];
        [encoder setBuffer:(id<MTLBuffer>)w offset:0 atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:y offset:output_offset atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15) / 16, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ++m->info.dispatches;
        return;
    }
    // Keep a one-candidate texture tail on the matrix path too. Scattered
    // scalar texture reads can dominate the entire final verification batch.
    int batched = tokens >= 8 || (m->verify_encoding && (tokens > 1 || packed == 3)),
        vector = tokens >= 4 && !batched;
    // Keep the minimum number of weight scans, then choose the narrowest
    // compiled tile: e.g. 69 tokens use 3 x 24 rather than 3 x 32.
    uint64_t panels = (tokens + 31) / 32;
    uint64_t tile_tokens = ((tokens + panels - 1) / panels + 7) / 8 * 8;
    int texture = packed == 3;
    unsigned direct_texture = texture && !offset && [(id<MTLTexture>)w width] * 4 == columns;
    // Direct-fragment kernels read each lane's BF16 words from complete texture
    // rows without staging. Up to 16 tokens read FP32 inputs in place; longer
    // inputs are arranged once per projection. The 8-column steps keep the
    // accumulation order of the prompt-matrix kernels.
    // Proposal prompts read Q4 codes the same way (more than 16 tokens only).
    int direct_q4 = packed == 2 && m->direct_q4_ready && tokens > 16 && !(columns % 32) &&
                    !(params.weight_offset % 32) && rows;
    if ((direct_texture && m->direct_ready && !(columns % 16) && rows && tokens && batched) ||
        direct_q4) {
        if (tokens <= 16) {
            unsigned wide = tokens > 8, rows_per_group = wide ? 64 : 128;
            [encoder setComputePipelineState:m->direct_raw[wide]];
            [encoder setTexture:(id<MTLTexture>)w atIndex:0];
            [encoder setBuffer:x offset:0 atIndex:1];
            [encoder setBuffer:y offset:output_offset atIndex:2];
            [encoder setBytes:&params length:sizeof params atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + rows_per_group - 1) / rows_per_group,
                                                      1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            return;
        }
        // Use the fewest panels of at most nine fragments, then the narrowest width.
        // A narrower final panel avoids padded fragments when it still has three.
        uint64_t fragments = (tokens + 7) / 8, panels = (fragments + 8) / 9;
        uint64_t width = (fragments + panels - 1) / panels, last = fragments - (panels - 1) * width;
        int tail = panels > 1 && last != width && last >= 3;
        uint64_t padded = tail ? fragments : panels * width;
        if (columns * padded * 32 <= m->arranged.length) {
            uint64_t key[3] = {tokens, columns, xs};
            if (!m->arranged_valid || m->arranged_input != x ||
                memcmp(key, m->arranged_key, sizeof key)) {
                struct {
                    uint32_t tokens, columns, input_stride, fragments;
                } arrange = {(uint32_t)tokens, (uint32_t)columns, (uint32_t)xs, (uint32_t)padded};
                uint64_t threads = columns / 8 * padded * 32;
                [encoder setComputePipelineState:m->arrange];
                [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:m->arranged offset:0 atIndex:1];
                [encoder setBytes:&arrange length:sizeof arrange atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                ++m->info.dispatches;
                m->arranged_valid = m->arranged_scope;
                m->arranged_input = x;
                memcpy(m->arranged_key, key, sizeof key);
            }
            params.xs = (uint32_t)padded;
            [encoder setComputePipelineState:direct_q4 ? m->direct_q4[width - 3]
                                                       : m->direct_texture[width - 3]];
            if (direct_q4)
                [encoder setBuffer:(id<MTLBuffer>)w offset:0 atIndex:0];
            else
                [encoder setTexture:(id<MTLTexture>)w atIndex:0];
            [encoder setBuffer:m->arranged offset:0 atIndex:1];
            [encoder setBuffer:y offset:output_offset atIndex:2];
            [encoder setBytes:&params length:sizeof params atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, panels - tail, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            if (tail) {
                params.first_fragment = (uint32_t)((panels - 1) * width);
                [encoder setComputePipelineState:direct_q4 ? m->direct_q4[last - 3]
                                                           : m->direct_texture[last - 3]];
                [encoder setBytes:&params length:sizeof params atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            return;
        }
    }
    if (texture && tokens > 32 && m->prompt_matrix_ready && !(columns % 16) &&
        params.aligned_weights && !(xs % 4)) {
        unsigned tile = (tokens + 63) / 64 * 64 == (tokens + 31) / 32 * 32 ? PROMPT_64 : PROMPT_32;
        // Just above 64 tokens, narrower panels reduce padded work. On M2,
        // wide projections favor three 24-token panels; others favor two 40s.
        if (tokens > 64 && tokens <= 80)
            tile = columns >= 8192 && tokens <= 72 ? PROMPT_24 : PROMPT_40;
        unsigned panel = prompt_tile_tokens[tile];
        [encoder setComputePipelineState:m->prompt_matrix[direct_texture][tile]];
        [encoder setTexture:(id<MTLTexture>)w atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:y offset:output_offset atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (tokens + panel - 1) / panel, 1)
                threadsPerThreadgroup:MTLSizeMake(tile == PROMPT_32 ? 256 : 128, 1, 1)];
        ++m->info.dispatches;
        return;
    }
    int compact = (m->verify_encoding && (!packed || texture) && tile_tokens <= 16) ||
                  (texture && tokens > 32);
    int texture_split = texture && compact && batched && tile_tokens == 8 && rows <= 16384 &&
                        columns >= 128 && !(columns % 128);
    if (texture_split || (m->split_k && batched && tokens <= 32 && columns >= 8192 &&
                          !(columns % 128) && rows <= 4096)) {
        params.ys = (uint32_t)rows;
        id<MTLComputePipelineState> pipeline =
            texture && compact && tile_tokens <= 16
                ? m->texture_split[direct_texture][tile_tokens / 8 - 1]
            : compact && tile_tokens == 16 ? m->compact_split
                                           : m->matmul_split[packed][tile_tokens / 8 - 1];
        if (texture_split && tokens == 8 && !(rows % 32) && params.aligned_weights && !(xs % 4))
            pipeline = m->full_texture[direct_texture];
        [encoder setComputePipelineState:pipeline];
        if (texture)
            [encoder setTexture:(id<MTLTexture>)w atIndex:0];
        else
            [encoder setBuffer:(id<MTLBuffer>)w offset:packed ? 0 : offset atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:m->split offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 31) / 32,
                                                  (tokens + tile_tokens - 1) / tile_tokens, 4)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        params.ys = (uint32_t)ys;
        [encoder setComputePipelineState:m->split_reduce];
        [encoder setBuffer:m->split offset:0 atIndex:0];
        [encoder setBuffer:y offset:output_offset atIndex:1];
        [encoder setBytes:&params length:sizeof params atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((rows * tokens + 127) / 128, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        m->info.dispatches += 2;
        return;
    }
    if (compact && batched) {
        unsigned rows_per_group = tile_tokens == 8 ? 32 : 64;
        [encoder
            setComputePipelineState:texture
                                        ? m->texture_compact[direct_texture][tile_tokens / 8 - 1]
                                        : m->compact[tile_tokens / 8 - 1]];
        if (texture)
            [encoder setTexture:(id<MTLTexture>)w atIndex:0];
        else
            [encoder setBuffer:(id<MTLBuffer>)w offset:offset atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:y offset:output_offset atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + rows_per_group - 1) / rows_per_group,
                                                  (tokens + tile_tokens - 1) / tile_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(rows_per_group * 4, 1, 1)];
        ++m->info.dispatches;
        return;
    }
    [encoder setComputePipelineState:batched  ? m->matmul[packed][tile_tokens / 8 - 1]
                                     : vector ? m->vector[packed]
                                              : m->matvec[packed]];
    if (texture)
        [encoder setTexture:(id<MTLTexture>)w atIndex:0];
    else
        [encoder setBuffer:(id<MTLBuffer>)w offset:packed ? 0 : offset atIndex:0];
    [encoder setBuffer:x offset:0 atIndex:1];
    [encoder setBuffer:y offset:output_offset atIndex:2];
    [encoder setBytes:&params length:sizeof params atIndex:3];
    MTLSize groups =
        batched ? MTLSizeMake((rows + 31) / 32, (tokens + tile_tokens - 1) / tile_tokens, 1)
                : MTLSizeMake((rows + 3) / 4, vector ? (tokens + 7) / 8 : tokens, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

int plamo_metal_complete(struct plamo_metal *m, id<MTLCommandBuffer> command,
                         id<MTLCommandEncoder> encoder) {
    [encoder endEncoding];
    // Batch initial weight registrations and new activation buffers. No GPU
    // work is submitted before the complete set of changes is committed.
    plamo_metal_residency_commit(m);
    [command commit];
    [command waitUntilCompleted];
    ++m->info.command_buffers;
    if (command.status != MTLCommandBufferStatusCompleted)
        return plamo_metal_problem(m, command.error ? command.error.localizedDescription.UTF8String
                                                    : "Metal command failed");
    m->info.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
    return 1;
}

int plamo_metal_linear(struct plamo_metal *m, const struct plamo_matmul *p) {
    if (!p->rows || !p->tokens)
        return 1;
    if (!p->columns) {
        for (uint64_t t = 0; t < p->tokens; ++t)
            memset(p->output + t * p->output_stride, 0, (size_t)p->rows * 4);
        return 1;
    }
    if (p->rows > UINT32_MAX || p->columns > UINT32_MAX || p->tokens > UINT32_MAX ||
        p->input_stride < p->columns || p->output_stride < p->rows ||
        p->rows > SIZE_MAX / 2 / p->columns || p->rows > SIZE_MAX / 4 / p->tokens ||
        p->columns > SIZE_MAX / 4 / p->tokens)
        return 0;
    @autoreleasepool {
        struct projection projection;
        int whole = p->rows * p->columns * 2 > m->staging_bytes;
        if (whole) {
            if (m->config.memory == PLAMO_METAL_AUTO && plamo_metal_now() - m->checked_at >= 0.25)
                plamo_metal_maintain(m);
            if (!plamo_metal_resolve_projection(m, p->weights, p->rows, p->columns, &projection,
                                                0)) {
                uint64_t window = m->staging_bytes / (p->columns * 2);
                if (window >= 8)
                    window = window / 8 * 8;
                if (!window)
                    return 0;
                // Streamed ranges finish before the read window is reused.
                // Resident projections instead encode every tile together.
                for (uint64_t first = 0; first < p->rows; first += window) {
                    struct plamo_matmul part = *p;
                    part.weights += first * p->columns;
                    part.output += first;
                    part.rows = p->rows - first;
                    if (part.rows > window)
                        part.rows = window;
                    if (!plamo_metal_linear(m, &part))
                        return 0;
                }
                return 1;
            }
        }
        if (!plamo_metal_ensure_buffer(m, &m->input, p->tokens * p->columns * 4) ||
            !plamo_metal_ensure_buffer(m, &m->output, p->tokens * p->rows * 4))
            return 0;
        NSUInteger offset = 0;
        unsigned packed = 0;
        id<MTLResource> w = whole ? nil
                                  : plamo_metal_weights(m, p->weights, p->rows * p->columns * 2,
                                                        &offset, 0, &packed, p->tokens == 1);
        if (!whole && !w)
            return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy((float *)m->input.contents + t * p->columns, p->input + t * p->input_stride,
                   (size_t)p->columns * 4);
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder)
            return plamo_metal_problem(m, "cannot create a Metal compute command");
        if (whole)
            plamo_metal_encode_projection(m, encoder, &projection, m->input, p->columns, m->output,
                                          p->rows, p->columns, p->tokens);
        else
            plamo_metal_encode_linear(m, encoder, w, offset, packed, m->input, p->columns,
                                      m->output, 0, p->rows, p->rows, p->columns, p->tokens);
        if (!plamo_metal_complete(m, command, encoder))
            return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy(p->output + t * p->output_stride, (float *)m->output.contents + t * p->rows,
                   (size_t)p->rows * 4);
        return 1;
    }
}

// Resolve the complete graph before changing model state. Uncached file views
// live in the enclosing autorelease pool until its GPU command is complete.
// Never refer to staging or evict a retained entry during command encoding.
int plamo_metal_mlp(struct plamo_metal *m, const struct mlp *p, float *states, uint64_t tokens) {
    m->verification_tokens = 0;
    if (!tokens)
        return 1;
    uint64_t h = p->hidden, n = p->intermediate;
    uint64_t norm_stride = (h * 2 + 15) & ~UINT64_C(15);
    if (!h || !n || h > UINT32_MAX || n > UINT32_MAX / 2 || tokens > UINT32_MAX ||
        tokens > SIZE_MAX / 4 / h || tokens > SIZE_MAX / 8 / n || h > SIZE_MAX / 4 / n)
        return 0;
    @autoreleasepool {
        if (m->config.memory == PLAMO_METAL_AUTO && plamo_metal_now() - m->checked_at >= 0.25)
            plamo_metal_maintain(m);
        struct projection gate, down;
        if (!plamo_metal_resolve_projection(m, p->gu, 2 * n, h, &gate, 0) ||
            !plamo_metal_resolve_projection(m, p->down, h, n, &down, 0))
            return 0;
        if (!plamo_metal_ensure_buffer(m, &m->state, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->normalized, tokens * h * 4) ||
            (m->draft_quantized && tokens == 1 &&
             !plamo_metal_ensure_buffer(m, &m->draft_normalized, h * 2)) ||
            !plamo_metal_ensure_buffer(m, &m->gate, tokens * n * 8) ||
            (m->draft_quantized && tokens == 1 &&
             !plamo_metal_ensure_buffer(m, &m->draft_gate, n * 2)) ||
            !plamo_metal_ensure_buffer(m, &m->branch, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->norms, norm_stride + h * 2))
            return 0;
        memcpy(m->state.contents, states, (size_t)(tokens * h * 4));
        memcpy(m->norms.contents, p->pre, (size_t)(h * 2));
        memcpy((char *)m->norms.contents + norm_stride, p->post, (size_t)(h * 2));
        struct {
            uint32_t hidden, intermediate, tokens;
        } params = {(uint32_t)h, (uint32_t)n, (uint32_t)tokens};
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder)
            return plamo_metal_problem(m, "cannot create a fused Metal MLP command");
        int auxiliary = m->draft_quantized && tokens == 1;
        [encoder setComputePipelineState:auxiliary ? m->draft_norm : m->norm];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:0 atIndex:1];
        [encoder setBuffer:m->normalized offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        if (auxiliary)
            [encoder setBuffer:m->draft_normalized offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        plamo_metal_encode_projection(m, encoder, &gate, m->normalized, h, m->gate, 2 * n, h,
                                      tokens);
        [encoder setComputePipelineState:auxiliary ? m->draft_swiglu : m->swiglu];
        [encoder setBuffer:m->gate offset:0 atIndex:0];
        [encoder setBytes:&params length:sizeof params atIndex:1];
        if (auxiliary)
            [encoder setBuffer:m->draft_gate offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        plamo_metal_encode_projection(m, encoder, &down, m->gate, 2 * n, m->branch, h, n, tokens);
        [encoder setComputePipelineState:m->residual];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:norm_stride atIndex:1];
        [encoder setBuffer:m->branch offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        m->info.dispatches += 3;
        if (!plamo_metal_complete(m, command, encoder))
            return 0;
        memcpy(states, m->state.contents, (size_t)(tokens * h * 4));
        ++m->info.fused_mlps;
        return 1;
    }
}

static int mlp(const struct mlp *p, float *states, uint64_t tokens, void *context) {
    return plamo_metal_mlp(context, p, states, tokens);
}
static int forward(struct model *model, uint64_t token, void *context) {
    return plamo_metal_forward(context, model, token);
}
static int prefill(struct model *model, const uint32_t *ids, uint64_t tokens, float *states,
                   void *context) {
    return plamo_metal_prefill(context, model, ids, tokens, states);
}

static void matmul(const struct plamo_matmul *p, void *context) {
    struct plamo_metal *m = context;
    if (!plamo_metal_linear(m, p)) {
        ++m->info.cpu_fallbacks;
        plamo_bf16_matmul_core(p);
    }
}
static void matvec(const uint16_t *w, const float *x, float *y, uint64_t rows, uint64_t columns,
                   void *context) {
    struct plamo_matmul p = {w, x, y, rows, columns, 1, columns, rows};
    matmul(&p, context);
}
// Defer original-byte acquisition until the GPU module can choose a retained
// shared buffer or its shared staging buffer. Never create a CPU staging copy.
static const void *acquire(const void *source, uint64_t bytes, void *context) {
    (void)bytes;
    (void)context;
    return source;
}

struct plamo_metal *plamo_metal_open(const struct plamo_metal_config *config, char *error,
                                     size_t cap) {
    @autoreleasepool {
        if (!config || !config->available || config->count > PLAMO_MAX_WEIGHT_FILES ||
            (unsigned)config->memory > PLAMO_METAL_STREAM ||
            config->precision != PLAMO_METAL_FP32 ||
            (unsigned)config->packing > PLAMO_PACK_DRAFT_Q4 || plamo_compute_context ||
            plamo_weight_provider) {
            snprintf(error, cap, "invalid Metal configuration or an existing compute owner");
            return NULL;
        }
        struct plamo_metal *m = calloc(1, sizeof *m);
        if (!m) {
            snprintf(error, cap, "cannot allocate Metal state");
            return NULL;
        }
        m->config = *config;
        m->draft_quantized =
            config->packing == PLAMO_PACK_DRAFT_Q4 && config->memory != PLAMO_METAL_STREAM;
        const char *private_weights = getenv("PLAMO_METAL_PRIVATE_WEIGHTS");
        m->private_weights = private_weights && !strcmp(private_weights, "1") &&
                             config->memory != PLAMO_METAL_STREAM;
        const char *texture_weights = getenv("PLAMO_METAL_TEXTURE_WEIGHTS");
        m->texture_weights = texture_weights && !strcmp(texture_weights, "1") &&
                             config->memory == PLAMO_METAL_AUTO && !m->draft_quantized;
        m->device = MTLCreateSystemDefaultDevice();
        if (!m->device || !m->device.hasUnifiedMemory ||
            ![m->device supportsFamily:MTLGPUFamilyApple7]) {
            snprintf(error, cap, "Metal requires an available Apple Silicon GPU");
            goto fail;
        }
        // Texture padding and both weight windows must fit before choosing a
        // fixed full-model cache. Smaller devices retain adaptive lossless IO.
        uint64_t preferred = config->working_weight_bytes;
        uint64_t texture_margin = preferred / 64 + 256 * MIB;
        if (!texture_weights && config->prefer_texture_weights &&
            config->packing == PLAMO_PACK_AUTO && config->memory == PLAMO_METAL_AUTO &&
            !m->private_weights && preferred &&
            preferred <= m->device.recommendedMaxWorkingSetSize &&
            texture_margin <= m->device.recommendedMaxWorkingSetSize - preferred)
            m->texture_weights = 1;
        snprintf(m->info.device, sizeof m->info.device, "%s", m->device.name.UTF8String);
        snprintf(m->info.precision, sizeof m->info.precision, "fp32");
        NSError *failure = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_3;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *))
            options.mathMode = MTLMathModeSafe;
        else {
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        }
#endif
        id<MTLLibrary> library =
            [m->device newLibraryWithSource:[NSString stringWithUTF8String:plamo_metal_source]
                                    options:options
                                      error:&failure];
        [options release];
        if (!library) {
            snprintf(error, cap, "Metal shader compilation failed: %s",
                     failure.localizedDescription.UTF8String);
            goto fail;
        }
        id<MTLFunction> fn;
        for (unsigned packed = 0; packed < (m->texture_weights   ? 4u
                                            : m->draft_quantized ? 3u
                                                                 : 2u);
             ++packed) {
            NSString *suffix = packed == 3   ? @"_texture"
                               : packed == 2 ? @"_q4"
                               : packed      ? @"_packed"
                                             : @"";
            fn = [library newFunctionWithName:[@"plamo_matvec" stringByAppendingString:suffix]];
            m->matvec[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            fn = [library
                newFunctionWithName:[@"plamo_matmul_vector" stringByAppendingString:suffix]];
            m->vector[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            NSString *names[] = {@"plamo_matmul8", @"plamo_matmul16", @"plamo_matmul_short",
                                 @"plamo_matmul"};
            for (unsigned n = 0; n < 4; ++n) {
                fn = [library newFunctionWithName:[names[n] stringByAppendingString:suffix]];
                m->matmul[packed][n] = [m->device newComputePipelineStateWithFunction:fn
                                                                                error:&failure];
                [fn release];
                fn = [library
                    newFunctionWithName:[NSString stringWithFormat:@"plamo_matmul%u_split%@",
                                                                   (n + 1) * 8, suffix]];
                m->matmul_split[packed][n] =
                    [m->device newComputePipelineStateWithFunction:fn error:&failure];
                [fn release];
            }
        }
        fn = [library newFunctionWithName:@"plamo_split_reduce"];
        m->split_reduce = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        // Optional: chained greedy steps fall back to single steps without them.
        fn = [library newFunctionWithName:@"plamo_draft_select"];
        if (fn)
            m->draft_select = [m->device newComputePipelineStateWithFunction:fn error:nil];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_draft_embed"];
        if (fn)
            m->draft_embed = [m->device newComputePipelineStateWithFunction:fn error:nil];
        [fn release];
        // The selection kernel adapts to fewer threads, e.g. under shader validation.
        if (m->draft_select && (m->draft_select.threadExecutionWidth != 32 ||
                                m->draft_select.maxTotalThreadsPerThreadgroup < 32)) {
            [m->draft_select release];
            m->draft_select = nil;
        }
        if (m->draft_embed && m->draft_embed.maxTotalThreadsPerThreadgroup < 32) {
            [m->draft_embed release];
            m->draft_embed = nil;
        }
        // Optional fused residual + RMSNorm for prompt and verification batches.
        // PLAMO_METAL_FUSE=0 keeps the separate kernels for comparison.
        const char *fuse = getenv("PLAMO_METAL_FUSE");
        fn =
            fuse && !strcmp(fuse, "0") ? nil : [library newFunctionWithName:@"plamo_residual_norm"];
        if (fn)
            m->residual_norm = [m->device newComputePipelineStateWithFunction:fn error:nil];
        [fn release];
        if (m->residual_norm && (m->residual_norm.threadExecutionWidth != 32 ||
                                 m->residual_norm.maxTotalThreadsPerThreadgroup < 128)) {
            [m->residual_norm release];
            m->residual_norm = nil;
        }
        if (m->texture_weights) {
            fn = [library newFunctionWithName:@"plamo_matrix_layout"];
            m->matrix_layout = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            for (unsigned direct = 0; direct < 2; ++direct)
                for (unsigned tile = 0; tile < PROMPT_TILE_COUNT; ++tile) {
                    fn = [library
                        newFunctionWithName:[NSString stringWithFormat:@"plamo_prompt%u_texture%@",
                                                                       prompt_tile_tokens[tile],
                                                                       direct ? @"_direct" : @""]];
                    MTLComputePipelineDescriptor *descriptor =
                        [[MTLComputePipelineDescriptor alloc] init];
                    descriptor.computeFunction = fn;
                    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                    descriptor.maxTotalThreadsPerThreadgroup = tile == PROMPT_32 ? 512 : 128;
                    m->prompt_matrix[direct][tile] =
                        [m->device newComputePipelineStateWithDescriptor:descriptor
                                                                 options:0
                                                              reflection:nil
                                                                   error:&failure];
                    [descriptor release];
                    [fn release];
                }
            for (unsigned i = 0; i < 9; ++i) {
                fn = [library
                    newFunctionWithName:i == 0 ? @"plamo_direct_raw8"
                                        : i == 1
                                            ? @"plamo_direct_raw16"
                                            : [NSString stringWithFormat:@"plamo_direct_texture%u",
                                                                         i + 1]];
                id<MTLComputePipelineState> pipeline = nil;
                if (fn) {
                    MTLComputePipelineDescriptor *descriptor =
                        [[MTLComputePipelineDescriptor alloc] init];
                    descriptor.computeFunction = fn;
                    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                    descriptor.maxTotalThreadsPerThreadgroup = 128;
                    pipeline = [m->device newComputePipelineStateWithDescriptor:descriptor
                                                                        options:0
                                                                     reflection:nil
                                                                          error:nil];
                    [descriptor release];
                    [fn release];
                }
                if (i < 2)
                    m->direct_raw[i] = pipeline;
                else
                    m->direct_texture[i - 2] = pipeline;
            }
            fn = [library newFunctionWithName:@"plamo_arrange_fragments"];
            if (fn)
                m->arrange = [m->device newComputePipelineStateWithFunction:fn error:nil];
            [fn release];
        }
        // Proposal prompts use direct Q4 fragments, which need the same layout
        // probe and input arrangement as the texture kernels.
        if (m->draft_quantized) {
            fn = [library newFunctionWithName:@"plamo_matrix_layout"];
            m->matrix_layout = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            fn = [library newFunctionWithName:@"plamo_arrange_fragments"];
            if (fn)
                m->arrange = [m->device newComputePipelineStateWithFunction:fn error:nil];
            [fn release];
            for (unsigned i = 0; i < 7; ++i) {
                fn = [library
                    newFunctionWithName:[NSString stringWithFormat:@"plamo_direct_q4_%u", i + 3]];
                if (fn) {
                    MTLComputePipelineDescriptor *descriptor =
                        [[MTLComputePipelineDescriptor alloc] init];
                    descriptor.computeFunction = fn;
                    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                    descriptor.maxTotalThreadsPerThreadgroup = 128;
                    m->direct_q4[i] = [m->device newComputePipelineStateWithDescriptor:descriptor
                                                                               options:0
                                                                            reflection:nil
                                                                                 error:nil];
                    [descriptor release];
                    [fn release];
                }
            }
        }
        for (unsigned n = 0; n < 2; ++n) {
            fn = [library newFunctionWithName:[NSString stringWithFormat:@"plamo_matmul%u_compact",
                                                                         (n + 1) * 8]];
            m->compact[n] = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
        }
        fn = [library newFunctionWithName:@"plamo_matmul16_split_compact"];
        m->compact_split = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        if (m->texture_weights)
            for (unsigned direct = 0; direct < 2; ++direct)
                for (unsigned n = 0; n < 4; ++n) {
                    NSString *suffix = direct ? @"_direct" : @"";
                    fn =
                        [library newFunctionWithName:
                                     [NSString stringWithFormat:@"plamo_matmul%u_compact_texture%@",
                                                                (n + 1) * 8, suffix]];
                    m->texture_compact[direct][n] =
                        [m->device newComputePipelineStateWithFunction:fn error:&failure];
                    [fn release];
                    if (n < 2) {
                        fn = [library
                            newFunctionWithName:
                                [NSString
                                    stringWithFormat:@"plamo_matmul%u_split_compact_texture%@",
                                                     (n + 1) * 8, suffix]];
                        m->texture_split[direct][n] =
                            [m->device newComputePipelineStateWithFunction:fn error:&failure];
                        [fn release];
                    }
                }
        if (m->texture_weights)
            for (unsigned direct = 0; direct < 2; ++direct) {
                fn = [library newFunctionWithName:direct ? @"plamo_matmul8_full_texture_direct"
                                                         : @"plamo_matmul8_full_texture"];
                m->full_texture[direct] = [m->device newComputePipelineStateWithFunction:fn
                                                                                   error:&failure];
                [fn release];
            }
        const char *split_k = getenv("PLAMO_METAL_SPLIT_K");
        m->split_k = split_k ? !strcmp(split_k, "4")
                             : m->texture_weights && config->working_weight_bytes > 8 * 1024 * MIB;
        fn = [library newFunctionWithName:@"plamo_mlp_norm"];
        m->norm = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_mlp_swiglu"];
        m->swiglu = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_mlp_swiglu_compact"];
        m->swiglu_compact = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        if (m->swiglu_compact.maxTotalThreadsPerThreadgroup < 128) {
            [m->swiglu_compact release];
            m->swiglu_compact = nil;
        }
        fn = [library newFunctionWithName:@"plamo_mlp_residual"];
        m->residual = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_cache_touch"];
        m->touch = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_pack_analyze"];
        m->analyze = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_pack_encode"];
        m->pack = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        if (m->draft_quantized) {
            fn = [library newFunctionWithName:@"plamo_draft_quantize4"];
            m->quantize4 = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            fn = [library newFunctionWithName:@"plamo_matvec_q4_rows"];
            m->q4_rows = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            for (unsigned i = 0; i < 2; ++i) {
                fn = [library newFunctionWithName:i ? @"plamo_draft_dot8" : @"plamo_draft_dot32"];
                m->draft_dot[i] = [m->device newComputePipelineStateWithFunction:fn error:&failure];
                [fn release];
            }
            // PLAMO_DRAFT_BLOCK=0 keeps the previous chunk-wise kernels for comparison.
            const char *block = getenv("PLAMO_DRAFT_BLOCK");
            for (unsigned i = 0; i < 2 && !(block && !strcmp(block, "0")); ++i) {
                fn = [library
                    newFunctionWithName:i ? @"plamo_draft_block4" : @"plamo_draft_block16"];
                if (fn)
                    m->draft_block[i] = [m->device newComputePipelineStateWithFunction:fn
                                                                                 error:nil];
                [fn release];
                // Fall back to the chunk-wise kernels if 128 threads cannot run.
                if (m->draft_block[i] && (m->draft_block[i].threadExecutionWidth != 32 ||
                                          m->draft_block[i].maxTotalThreadsPerThreadgroup < 128)) {
                    [m->draft_block[i] release];
                    m->draft_block[i] = nil;
                }
            }
            fn = [library newFunctionWithName:@"plamo_draft_norm"];
            m->draft_norm = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            fn = [library newFunctionWithName:@"plamo_draft_swiglu"];
            m->draft_swiglu = [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            // Optional fused residual + RMSNorm for single-token proposals.
            // PLAMO_DRAFT_FUSE=0 keeps the separate kernels for comparison.
            const char *fuse = getenv("PLAMO_DRAFT_FUSE");
            fn = fuse && !strcmp(fuse, "0")
                     ? nil
                     : [library newFunctionWithName:@"plamo_draft_residual_norm"];
            if (fn)
                m->draft_residual_norm = [m->device newComputePipelineStateWithFunction:fn
                                                                                  error:nil];
            [fn release];
            if (m->draft_residual_norm &&
                (m->draft_residual_norm.threadExecutionWidth != 32 ||
                 m->draft_residual_norm.maxTotalThreadsPerThreadgroup < 128)) {
                [m->draft_residual_norm release];
                m->draft_residual_norm = nil;
            }
        }
        fn = [library newFunctionWithName:@"plamo_decode_keys"];
        m->decode_keys = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_qkv"];
        m->decode_qkv = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_attention"];
        m->decode_attention = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_residual"];
        m->decode_residual = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_qkv"];
        m->prefill_qkv = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_qkv8"];
        m->prefill_qkv8 = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_attention"];
        m->prefill_attention = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        NSString *attention_names[] = {@"plamo_prompt_attention128", @"plamo_prompt_attention512",
                                       @"plamo_prompt_attention_matrix256",
                                       @"plamo_prompt_attention_matrix512"};
        for (unsigned i = 0; i < 4; ++i) {
            fn = [library newFunctionWithName:attention_names[i]];
            id<MTLComputePipelineState> pipeline =
                [m->device newComputePipelineStateWithFunction:fn error:&failure];
            [fn release];
            if ((i < 2 || pipeline.threadExecutionWidth == 32) &&
                pipeline.maxTotalThreadsPerThreadgroup >= (i >= 2 ? 256 : 128) &&
                pipeline.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength)
                m->prompt_attention[i] = pipeline;
            else
                [pipeline release];
        }
        fn = [library newFunctionWithName:@"plamo_prefill_store"];
        m->prefill_store = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_verify_scores"];
        m->verify_scores = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_verify_softmax"];
        m->verify_softmax = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_verify_values"];
        m->verify_values = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        [library release];
        m->queue = [m->device newCommandQueue];
        if (m->matrix_layout && m->queue) {
            id<MTLBuffer> probe = [m->device newBufferWithLength:64 * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [m->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (probe && command && encoder) {
                [encoder setComputePipelineState:m->matrix_layout];
                [encoder setBuffer:probe offset:0 atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                int layout = command.status == MTLCommandBufferStatusCompleted;
                const float *values = probe.contents;
                for (unsigned lane = 0; layout && lane < 32; ++lane) {
                    unsigned row = ((lane & 6) >> 1) | ((lane & 16) >> 2);
                    unsigned col = ((lane & 1) << 1) | ((lane & 8) >> 1);
                    layout = values[lane * 2] == row * 8 + col &&
                             values[lane * 2 + 1] == row * 8 + col + 1;
                }
                int matches = layout;
                for (unsigned d = 0; d < 2; ++d)
                    for (unsigned n = 0; n < PROMPT_TILE_COUNT; ++n) {
                        id<MTLComputePipelineState> p = m->prompt_matrix[d][n];
                        matches =
                            matches && p && p.threadExecutionWidth == 32 &&
                            p.maxTotalThreadsPerThreadgroup >= (n == PROMPT_32 ? 256u : 128u) &&
                            p.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength;
                    }
                m->prompt_matrix_ready = matches;
                // The direct kernels use the same fragment layout. Setting
                // PLAMO_METAL_DIRECT=0 keeps the previous texture kernels.
                const char *direct = getenv("PLAMO_METAL_DIRECT");
                int arranging = !(direct && !strcmp(direct, "0")) && m->arrange &&
                                m->arrange.threadExecutionWidth == 32 &&
                                m->arrange.maxTotalThreadsPerThreadgroup >= 256;
                int ready = matches && arranging;
                for (unsigned i = 0; i < 9; ++i) {
                    id<MTLComputePipelineState> p =
                        i < 2 ? m->direct_raw[i] : m->direct_texture[i - 2];
                    ready = ready && p && p.threadExecutionWidth == 32 &&
                            p.maxTotalThreadsPerThreadgroup >= 128;
                }
                m->direct_ready = ready;
                int q4 = m->draft_quantized && layout && arranging;
                for (unsigned i = 0; i < 7; ++i)
                    q4 = q4 && m->direct_q4[i] && m->direct_q4[i].threadExecutionWidth == 32 &&
                         m->direct_q4[i].maxTotalThreadsPerThreadgroup >= 128;
                m->direct_q4_ready = q4;
            } else
                [encoder endEncoding];
            [probe release];
        }
        int pipelines = (!m->draft_quantized ||
                         (m->quantize4 && m->q4_rows && m->q4_rows.threadExecutionWidth == 32 &&
                          m->q4_rows.maxTotalThreadsPerThreadgroup >= 128)) &&
                        m->norm && m->swiglu && m->residual && m->touch && m->analyze && m->pack &&
                        m->queue && m->split_reduce && m->split_reduce.threadExecutionWidth == 32 &&
                        m->split_reduce.maxTotalThreadsPerThreadgroup >= 128 && m->decode_keys &&
                        m->decode_qkv && m->decode_attention && m->decode_residual &&
                        m->prefill_qkv && m->prefill_attention && m->prefill_store &&
                        m->verify_scores && m->verify_softmax && m->verify_values;
        for (unsigned packed = 0; packed < (m->texture_weights   ? 4u
                                            : m->draft_quantized ? 3u
                                                                 : 2u);
             ++packed) {
            pipelines = pipelines && m->matvec[packed] && m->vector[packed] &&
                        m->matvec[packed].threadExecutionWidth == 32 &&
                        m->vector[packed].threadExecutionWidth == 32 &&
                        m->matvec[packed].maxTotalThreadsPerThreadgroup >= 128 &&
                        m->vector[packed].maxTotalThreadsPerThreadgroup >= 128;
            for (unsigned n = 0; n < 4; ++n) {
                id<MTLComputePipelineState> p = m->matmul[packed][n];
                id<MTLComputePipelineState> split = m->matmul_split[packed][n];
                pipelines =
                    pipelines && p && p.threadExecutionWidth == 32 &&
                    p.maxTotalThreadsPerThreadgroup >= 128 &&
                    p.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength &&
                    split && split.threadExecutionWidth == 32 &&
                    split.maxTotalThreadsPerThreadgroup >= 128 &&
                    split.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength;
            }
        }
        id<MTLComputePipelineState> compact_pipelines[] = {m->compact[0], m->compact[1],
                                                           m->compact_split};
        for (unsigned i = 0; i < 3; ++i) {
            id<MTLComputePipelineState> p = compact_pipelines[i];
            pipelines = pipelines && p && p.threadExecutionWidth == 32 &&
                        p.maxTotalThreadsPerThreadgroup >= (i == 1 ? 256u : 128u) &&
                        p.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength;
        }
        if (m->texture_weights)
            for (unsigned direct = 0; direct < 2; ++direct)
                for (unsigned i = 0; i < 4; ++i) {
                    id<MTLComputePipelineState> p = m->texture_compact[direct][i];
                    pipelines =
                        pipelines && p && p.threadExecutionWidth == 32 &&
                        p.maxTotalThreadsPerThreadgroup >= (i ? 256u : 128u) &&
                        p.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength;
                    if (i < 2) {
                        id<MTLComputePipelineState> split = m->texture_split[direct][i];
                        pipelines = pipelines && split && split.threadExecutionWidth == 32 &&
                                    split.maxTotalThreadsPerThreadgroup >= 128 &&
                                    split.staticThreadgroupMemoryLength <=
                                        m->device.maxThreadgroupMemoryLength;
                    }
                }
        id<MTLComputePipelineState> extra[] = {m->draft_dot[0], m->draft_dot[1], m->draft_norm,
                                               m->draft_swiglu};
        if (m->draft_quantized)
            for (unsigned i = 0; i < 4; ++i)
                pipelines = pipelines && extra[i] && extra[i].threadExecutionWidth == 32 &&
                            extra[i].maxTotalThreadsPerThreadgroup >= 128;
        if (m->texture_weights)
            for (unsigned i = 0; i < 2; ++i)
                pipelines = pipelines && m->full_texture[i] &&
                            m->full_texture[i].threadExecutionWidth == 32 &&
                            m->full_texture[i].maxTotalThreadsPerThreadgroup >= 128 &&
                            m->full_texture[i].staticThreadgroupMemoryLength <=
                                m->device.maxThreadgroupMemoryLength;
        if (!pipelines) {
            snprintf(error, cap, "Metal compute pipeline is unavailable: %s",
                     failure ? failure.localizedDescription.UTF8String : "unsupported limits");
            goto fail;
        }
        uint64_t total = 0;
        for (size_t i = 0; i < config->count; ++i) {
            if (!config->maps || !config->fds || !config->maps[i].base || !config->maps[i].bytes ||
                config->fds[i] < 0 || config->fds[i] > INT_MAX ||
                UINT64_MAX - total < config->maps[i].bytes) {
                snprintf(error, cap, "invalid Metal weight mapping");
                goto fail;
            }
            total += config->maps[i].bytes;
        }
        uint64_t available = config->available();
        uint64_t spare = available > config->reserve ? available - config->reserve : 0;
        m->staging_bytes = spare / 8;
        uint64_t window = m->texture_weights && config->working_weight_bytes > 8 * 1024 * MIB
                              ? 128 * MIB
                              : 64 * MIB;
        if (m->staging_bytes > window)
            m->staging_bytes = window;
        m->staging_bytes = m->staging_bytes / 16384 * 16384;
        if (m->staging_bytes < MIB)
            m->staging_bytes = MIB;
        uint64_t device_budget = m->device.recommendedMaxWorkingSetSize;
        if (device_budget > 128 * MIB)
            device_budget -= 128 * MIB;
        m->maximum = spare < device_budget ? spare : device_budget;
        // Keep room for the page padding of individual retained ranges. The
        // budget is an upper bound, not an allocation of the whole amount.
        if (m->maximum < m->staging_bytes)
            m->maximum = m->staging_bytes;
        m->capacity = config->memory == PLAMO_METAL_STREAM ? m->staging_bytes : m->maximum;
        uint64_t working = config->working_weight_bytes ? config->working_weight_bytes : total;
        m->use_packing =
            config->memory != PLAMO_METAL_STREAM &&
            (config->packing == PLAMO_PACK_LOSSLESS ||
             (config->packing == PLAMO_PACK_AUTO && working > m->capacity - m->staging_bytes));
        const char *fixed_bf16 = getenv("PLAMO_METAL_FIXED_BF16");
        m->fixed_bf16 =
            m->texture_weights || (fixed_bf16 && !strcmp(fixed_bf16, "1") &&
                                   config->memory == PLAMO_METAL_AUTO && !m->draft_quantized);
        if (m->fixed_bf16) {
            if (device_budget < m->staging_bytes || working > device_budget - m->staging_bytes) {
                snprintf(error, cap, "fixed BF16 weights exceed the Metal working set");
                goto fail;
            }
            m->capacity = m->maximum = device_budget;
            m->use_packing = 0;
        }
        const char *texture_heap = getenv("PLAMO_METAL_TEXTURE_HEAP");
        if (m->texture_weights && texture_heap && !strcmp(texture_heap, "1")) {
            uint64_t margin = working / 64 + 65536;
            uint64_t budget = m->capacity - m->staging_bytes;
            uint64_t wanted = working > UINT64_MAX - margin ? budget : working + margin;
            m->texture_heap_budget = (wanted < budget ? wanted : budget) / 16384 * 16384;
        }
        const char *raw_mappings = getenv("PLAMO_METAL_RAW_MMAP");
        m->raw_mappings = !m->fixed_bf16 && !m->private_weights && raw_mappings &&
                          !strcmp(raw_mappings, "1") && config->memory != PLAMO_METAL_STREAM &&
                          !m->draft_quantized;
        const char *raw_residency = getenv("PLAMO_METAL_RAW_RESIDENCY");
        m->raw_residency = m->raw_mappings && raw_residency && !strcmp(raw_residency, "1");
        if (m->raw_mappings)
            m->use_packing = 0;
        if (config->memory == PLAMO_METAL_RESIDENT &&
            ((!m->use_packing && !m->draft_quantized && total > spare) ||
             (m->use_packing && total / 2 > m->capacity - m->staging_bytes))) {
            snprintf(error, cap, "Metal BF16 preload requires weights plus the memory reserve");
            goto fail;
        }
        m->staging = [m->device newBufferWithLength:(NSUInteger)m->staging_bytes
                                            options:MTLResourceStorageModeShared];
        if (!m->staging) {
            snprintf(error, cap, "cannot allocate a shared Metal weight window");
            goto fail;
        }
        if ((m->split_k || m->texture_weights) &&
            !plamo_metal_ensure_buffer(m, &m->split, 2 * MIB)) {
            snprintf(error, cap, "cannot allocate split-K workspace");
            goto fail;
        }
        // Arranged FP32 prompt inputs: 16384 columns by 512 tokens (8B), or
        // 6144 columns by about 1300 tokens for proposal prompts.
        if (m->direct_ready || m->direct_q4_ready) {
            m->arranged = [m->device newBufferWithLength:32 * MIB
                                                 options:MTLResourceStorageModePrivate];
            if (!m->arranged)
                m->direct_ready = m->direct_q4_ready = 0;
        }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *descriptor = [[MTLResidencySetDescriptor alloc] init];
            descriptor.initialCapacity = MAX_ENTRIES + 10;
            m->residency = [m->device newResidencySetWithDescriptor:descriptor error:&failure];
            [descriptor release];
            if (!m->residency) {
                snprintf(error, cap, "cannot create Metal residency set: %s",
                         failure ? failure.localizedDescription.UTF8String : "unavailable");
                goto fail;
            }
            [m->queue addResidencySet:m->residency];
            plamo_metal_residency_add(m, m->staging);
            plamo_metal_residency_add(m, m->arranged);
            plamo_metal_residency_commit(m);
            [m->residency requestResidency];
            m->info.residency_sets = 1;
        }
#endif
        if (config->disk_cache_path && config->memory != PLAMO_METAL_STREAM &&
            !m->draft_quantized && !m->raw_mappings && !m->fixed_bf16) {
            m->disk_cache =
                plamo_disk_cache_open(config->disk_cache_path, config->revision, config->maps,
                                      config->count, m->info.note, sizeof m->info.note);
            if (config->packing == PLAMO_PACK_AUTO && plamo_disk_cache_packed(m->disk_cache))
                m->use_packing = 1;
        }
        m->config.disk_cache_path = NULL; // The path is borrowed only during open.
        for (size_t i = 0; i < config->count; ++i)
            (void)fcntl((int)config->fds[i], F_NOCACHE, 1);
        m->checked_at = plamo_metal_now();
        if (config->memory == PLAMO_METAL_RESIDENT && !m->raw_mappings &&
            !plamo_metal_preload_schema(m)) {
            snprintf(error, cap, "%s", m->info.note);
            goto fail;
        }
        plamo_weight_context = m;
        plamo_weight_provider = acquire;
        // The GPU module owns row tiling and combines resident tiles into one
        // command. The assembly caller therefore submits complete projections.
        plamo_weight_chunk_bytes = 0;
        plamo_compute_context = m;
        plamo_matvec_kernel = matvec;
        plamo_matmul_kernel = matmul;
        plamo_mlp_kernel = mlp;
        plamo_forward_kernel = forward;
        plamo_prefill_kernel = prefill;
        return m;
    fail:
        plamo_metal_close(m);
        return NULL;
    }
}

void plamo_metal_get_info(const struct plamo_metal *m, struct plamo_metal_info *out) {
    *out = m->info;
    out->cache.budget_bytes = m->capacity;
    out->cache.max_budget_bytes = m->maximum;
    out->cache.staging_bytes = m->staging_bytes;
    out->cache.storage_bytes = m->used;
    snprintf(out->weight_storage, sizeof out->weight_storage, "%s",
             m->draft_quantized                                               ? "q4-draft"
             : m->texture_weights                                             ? "texture-bf16"
             : m->raw_mappings                                                ? "mapped-bf16"
             : m->info.packed_weight_bytes || m->info.disk_cache_mapped_bytes ? "lossless-bf16"
                                                                              : "bf16");
    out->allocated_bytes = m->device.currentAllocatedSize;
    out->weight_heap_bytes = m->texture_heap && m->texture_heap.usedSize ? m->texture_heap.size : 0;
    out->recommended_working_set_bytes = m->device.recommendedMaxWorkingSetSize;
}

void plamo_metal_close(struct plamo_metal *m) {
    if (!m)
        return;
    if (plamo_compute_context == m) {
        plamo_matvec_kernel = NULL;
        plamo_matmul_kernel = NULL;
        plamo_mlp_kernel = NULL;
        plamo_compute_context = NULL;
        plamo_forward_kernel = NULL;
        plamo_prefill_kernel = NULL;
        plamo_weight_provider = NULL;
        plamo_weight_context = NULL;
        plamo_weight_chunk_bytes = 0;
    }
    plamo_metal_release_decode(m);
    plamo_metal_release_mappings(m);
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (m->residency) {
            [m->queue removeResidencySet:m->residency];
            [m->residency endResidency];
            [m->residency removeAllAllocations];
            [m->residency commit];
            [m->residency release];
        }
    }
#endif
    for (size_t i = 0; i < m->entries; ++i)
        [m->table[i].resource release];
    [m->texture_heap release];
    [m->staging release];
    [m->input release];
    [m->output release];
    [m->state release];
    [m->normalized release];
    [m->gate release];
    [m->branch release];
    [m->norms release];
    for (unsigned packed = 0; packed < (m->texture_weights   ? 4u
                                        : m->draft_quantized ? 3u
                                                             : 2u);
         ++packed) {
        [m->matvec[packed] release];
        [m->vector[packed] release];
        for (unsigned n = 0; n < 4; ++n) {
            [m->matmul[packed][n] release];
            [m->matmul_split[packed][n] release];
        }
    }
    [m->norm release];
    [m->swiglu release];
    [m->swiglu_compact release];
    [m->residual release];
    [m->touch release];
    [m->pack_headers release];
    [m->analyze release];
    [m->pack release];
    [m->quantize4 release];
    [m->decode_keys release];
    [m->decode_qkv release];
    [m->decode_attention release];
    [m->decode_residual release];
    [m->prefill_qkv release];
    [m->prefill_qkv8 release];
    [m->prefill_attention release];
    for (unsigned i = 0; i < 4; ++i)
        [m->prompt_attention[i] release];
    [m->prefill_store release];
    [m->verify_scores release];
    [m->verify_softmax release];
    [m->verify_values release];
    [m->decode_norms release];
    [m->qkv release];
    [m->attention release];
    [m->scores release];
    [m->logits release];
    [m->split release];
    [m->split_reduce release];
    [m->compact[0] release];
    [m->compact[1] release];
    [m->compact_split release];
    [m->matrix_layout release];
    for (unsigned d = 0; d < 2; ++d)
        for (unsigned n = 0; n < PROMPT_TILE_COUNT; ++n)
            [m->prompt_matrix[d][n] release];
    for (unsigned d = 0; d < 2; ++d)
        for (unsigned n = 0; n < 4; ++n) {
            [m->texture_compact[d][n] release];
            if (n < 2)
                [m->texture_split[d][n] release];
        }
    [m->q4_rows release];
    [m->draft_dot[0] release];
    [m->draft_dot[1] release];
    [m->draft_norm release];
    [m->draft_swiglu release];
    [m->draft_residual_norm release];
    [m->residual_norm release];
    [m->verify_results release];
    [m->draft_block[0] release];
    [m->draft_block[1] release];
    [m->draft_normalized release];
    [m->draft_gate release];
    [m->full_texture[0] release];
    [m->full_texture[1] release];
    for (unsigned i = 0; i < 2; ++i)
        [m->direct_raw[i] release];
    for (unsigned i = 0; i < 7; ++i) {
        [m->direct_texture[i] release];
        [m->direct_q4[i] release];
    }
    [m->arrange release];
    [m->arranged release];
    [m->draft_select release];
    [m->draft_embed release];
    [m->draft_results release];
    [m->draft_embedding release];
    [m->queue release];
    [m->device release];
    plamo_disk_cache_close(m->disk_cache);
    for (size_t i = 0; i < m->config.count; ++i)
        if (m->config.fds && m->config.fds[i] >= 0 && m->config.fds[i] <= INT_MAX)
            (void)fcntl((int)m->config.fds[i], F_NOCACHE, 0);
    free(m);
}
