#include "metal_internal.h"
#include <stdio.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
extern void plamo_rope_table(const float *, uint64_t, uint64_t, float *, float *, float);

static int prepare_graph(struct plamo_metal *m) {
    struct projection *projections[4 * 128 + 1];
    size_t count = 0;
    for (size_t i = 0; i < m->layer_count; ++i) {
        struct decode_layer *d = &m->layers[i];
        projections[count++] = &d->qkv;
        projections[count++] = &d->attention;
        projections[count++] = &d->gate;
        projections[count++] = &d->down;
    }
    projections[count++] = &m->head;
    // Check the whole file-backed working set before creating any views. A
    // model larger than the device recommendation keeps the row-wise path.
    uint64_t limit = m->device.recommendedMaxWorkingSetSize;
    if (limit > 128 * MIB)
        limit -= 128 * MIB;
    uint64_t bytes = m->used + m->staging_bytes;
    for (size_t i = 0; i < count; ++i) {
        struct projection *p = projections[i];
        for (size_t j = 0; j < p->count; ++j) {
            uint64_t n = p->tiles[j].mapped_bytes;
            if (n && (bytes > limit || n > limit - bytes))
                return 0;
            bytes += n;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        struct projection *p = projections[i];
        for (size_t j = 0; j < p->count; ++j) {
            struct tile *t = &p->tiles[j];
            if (t->mapped_data &&
                !(t->resource = plamo_metal_mapped_weights(m, t->mapped_data, t->mapped_bytes)))
                return 0;
        }
    }
    return 1;
}

static void release_view(struct plamo_metal *m, struct view *v) {
    if (v->buffer) {
        plamo_metal_residency_remove(m, v->buffer);
        [v->buffer release];
    }
    *v = (struct view){0};
}

// Map only a growing prefix of the existing anonymous model arena. The full
// global KV capacity is virtual (262K rows); making it all resident would
// displace model weights. No copy or alternate logical KV cache is needed.
static int cache_view(struct plamo_metal *m, struct view *v, void *source, uint64_t rows,
                      uint64_t width, uint64_t capacity) {
    if (!source || !rows || !width || rows > capacity || rows > SIZE_MAX / 4 / width)
        return 0;
    uint64_t bytes = rows * width * 4;
    if (v->source == source && v->bytes >= bytes)
        return 1;
    uint64_t size = 128;
    while (size < rows && size <= UINT64_MAX / 2)
        size *= 2;
    if (size > capacity)
        size = capacity;
    if (size > SIZE_MAX / 4 / width)
        return 0;
    bytes = size * width * 4;
    uintptr_t address = (uintptr_t)source;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return 0;
    uint64_t page = (uint64_t)page_size, offset = address % page;
    if (bytes > UINT64_MAX - offset - page)
        return 0;
    uint64_t length = (bytes + offset + page - 1) / page * page;
    if (length > m->device.maxBufferLength)
        return 0;
    mach_vm_address_t base = address - offset;
    if (base > UINT64_MAX - length)
        return 0;
    // Sharing/touching a prefix may split one anonymous arena into adjacent
    // VM regions. Validate every covered region, rather than requiring the
    // larger view to fit the first region left by an earlier shorter view.
    for (mach_vm_address_t cursor = base; cursor < base + length;) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t region_size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t status =
            mach_vm_region(mach_task_self(), &region, &region_size, VM_REGION_BASIC_INFO_64,
                           (vm_region_info_t)&info, &count, &object);
        if (object != MACH_PORT_NULL)
            mach_port_deallocate(mach_task_self(), object);
        if (status != KERN_SUCCESS || region > cursor || region_size > UINT64_MAX - region ||
            region + region_size <= cursor ||
            (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) != (VM_PROT_READ | VM_PROT_WRITE))
            return 0;
        cursor = region + region_size;
    }
    id<MTLBuffer> replacement = [m->device newBufferWithBytesNoCopy:(void *)base
                                                             length:(NSUInteger)length
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
    if (!replacement)
        return 0;
    release_view(m, v);
    *v = (struct view){replacement, source, (NSUInteger)offset, bytes};
    plamo_metal_residency_add(m, replacement);
    plamo_metal_residency_commit(m);
    return 1;
}

void plamo_metal_release_decode(struct plamo_metal *m) {
    m->verification_tokens = 0;
    for (uint64_t i = 0; i < m->layer_count; ++i) {
        release_view(m, &m->layers[i].k);
        release_view(m, &m->layers[i].v);
        release_view(m, &m->layers[i].rotated);
        plamo_metal_residency_remove(m, m->layers[i].proposal);
        [m->layers[i].proposal release];
        plamo_metal_residency_remove(m, m->layers[i].proposal_rotated);
        [m->layers[i].proposal_rotated release];
    }
    for (uint64_t i = 0; i < m->rope_count; ++i) {
        plamo_metal_residency_remove(m, m->ropes[i].buffer);
        [m->ropes[i].buffer release];
    }
    free(m->layers);
    free(m->ropes);
    m->layers = NULL;
    m->ropes = NULL;
    m->model = NULL;
    m->layer_count = m->rope_count = 0;
    plamo_metal_residency_commit(m);
}

static int prepare_rope(struct plamo_metal *m, struct rope_table *r, uint64_t positions) {
    if (positions <= r->positions)
        return 1;
    uint64_t rows = 128;
    while (rows < positions)
        rows *= 2;
    if (rows > UINT32_MAX || r->dim > SIZE_MAX / 4 / rows)
        return 0;
    uint64_t previous = r->positions;
    id<MTLBuffer> replacement =
        [m->device newBufferWithLength:(NSUInteger)plamo_metal_aligned(rows * r->dim * 4)
                               options:MTLResourceStorageModeShared];
    if (!replacement)
        return plamo_metal_problem(m, "cannot allocate exact RoPE table");
    if (previous)
        memcpy(replacement.contents, r->buffer.contents, (size_t)(previous * r->dim * 4));
    float scale;
    memcpy(&scale, &r->scale, 4);
    for (uint64_t row = previous; row < rows; ++row) {
        float *p = (float *)replacement.contents + row * r->dim;
        plamo_rope_table(r->frequency, row, r->dim / 2, p, p + r->dim / 2, scale);
    }
    if (r->buffer) {
        plamo_metal_residency_remove(m, r->buffer);
        [r->buffer release];
    }
    r->buffer = replacement;
    r->positions = rows;
    plamo_metal_residency_add(m, replacement);
    return 1;
}

static int prepare_verification_scratch(struct plamo_metal *m, const struct model *model) {
    if (!m->texture_weights || m->config.working_weight_bytes <= 8 * 1024 * MIB)
        return 1;
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct qkv *q = model->layers[i]->mixer->qkv;
        struct decode_layer *d = &m->layers[i];
        if (!plamo_metal_ensure_buffer(m, &d->proposal,
                                       PLAMO_VERIFY_MAX * (q->qh + 2 * q->kh) * q->dim * 4) ||
            !plamo_metal_ensure_buffer(m, &d->proposal_rotated,
                                       PLAMO_VERIFY_MAX * q->kh * q->dim * 4))
            return 0;
    }
    if (!plamo_metal_ensure_buffer(m, &m->logits, PLAMO_VERIFY_MAX * model->vocab * 4) ||
        !plamo_metal_ensure_buffer(m, &m->scores, 16 * MIB))
        return 0;
    id<MTLCommandBuffer> command = [m->queue commandBuffer];
    id<MTLBlitCommandEncoder> e = [command blitCommandEncoder];
    if (!command || !e)
        return plamo_metal_problem(m, "cannot prepare verification scratch");
    for (uint64_t i = 0; i < model->count; ++i) {
        struct decode_layer *d = &m->layers[i];
        [e fillBuffer:d->proposal range:NSMakeRange(0, d->proposal.length) value:0];
        [e fillBuffer:d->proposal_rotated range:NSMakeRange(0, d->proposal_rotated.length) value:0];
    }
    [e fillBuffer:m->logits range:NSMakeRange(0, m->logits.length) value:0];
    [e fillBuffer:m->scores range:NSMakeRange(0, m->scores.length) value:0];
    return plamo_metal_complete(m, command, e);
}

int plamo_metal_prepare_model(struct plamo_metal *m, struct model *model) {
    if (m->config.memory == PLAMO_METAL_STREAM || m->model == model)
        return 1;
    if (!model || !model->count || model->count > 128 || !model->hidden ||
        model->hidden > UINT32_MAX || !model->vocab || model->vocab > UINT32_MAX)
        return 0;
    @autoreleasepool {
        plamo_metal_release_decode(m);
        m->layers = calloc((size_t)model->count, sizeof *m->layers);
        m->ropes = calloc((size_t)model->count, sizeof *m->ropes);
        if (!m->layers || !m->ropes) {
            plamo_metal_release_decode(m);
            return plamo_metal_problem(m, "cannot allocate decode descriptors");
        }
        m->layer_count = model->count;
        uint64_t norm_bytes = 0, qkv = 0, attention = 0, gate = 0, heads = 0;
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            if (l->hidden != model->hidden || q->hidden != model->hidden ||
                l->mlp->hidden != model->hidden || !q->kh || !q->qh || q->kh > UINT32_MAX ||
                q->qh > UINT32_MAX || q->qh % q->kh || !q->dim || q->dim % 2 ||
                q->dim > UINT32_MAX / (q->qh + 2 * q->kh) || !c->capacity ||
                c->capacity > UINT32_MAX || c->width != q->kh * q->dim || r->cache != c ||
                r->heads != q->kh || r->dim != q->dim || !l->mlp->intermediate ||
                l->mlp->intermediate > UINT32_MAX / 2) {
                plamo_metal_release_decode(m);
                return 0;
            }
            struct decode_layer *d = &m->layers[i];
            NSUInteger *offsets[] = {&d->pre, &d->post, &d->mlp_pre, &d->mlp_post, &d->qn, &d->kn};
            for (size_t j = 0; j < 6; ++j) {
                *offsets[j] = (NSUInteger)norm_bytes;
                norm_bytes += ((j < 4 ? model->hidden : q->dim) * 2 + 15) & ~UINT64_C(15);
            }
            uint32_t scale;
            memcpy(&scale, &model->scales[i], 4);
            size_t rope = 0;
            for (; rope < m->rope_count; ++rope) {
                struct rope_table *t = &m->ropes[rope];
                if (t->dim == q->dim && t->scale == scale &&
                    !memcmp(t->frequency, r->freq, (size_t)(q->dim * 2)))
                    break;
            }
            if (rope == m->rope_count)
                m->ropes[m->rope_count++] = (struct rope_table){r->freq, q->dim, 0, scale, nil};
            d->rope = rope;
            // API accepts up to 4096 prompt + 4096 output tokens. Grow further
            // only if another native caller actually needs a later position.
            if (!prepare_rope(m, &m->ropes[rope], c->capacity < 8192 ? c->capacity : 8192) ||
                !cache_view(m, &d->k, c->k, 1, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, 1, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, r->out, 1, c->width, c->capacity)) {
                plamo_metal_release_decode(m);
                return 0;
            }
            uint64_t n = (q->qh + 2 * q->kh) * q->dim;
            if (n > qkv)
                qkv = n;
            if (q->qh * q->dim > attention)
                attention = q->qh * q->dim;
            if (q->qh > heads)
                heads = q->qh;
            if (l->mlp->intermediate * 2 > gate)
                gate = l->mlp->intermediate * 2;
        }
        m->final_norm = (NSUInteger)norm_bytes;
        norm_bytes += model->hidden * 2;
        if (!plamo_metal_ensure_buffer(m, &m->decode_norms, norm_bytes) ||
            !plamo_metal_ensure_buffer(m, &m->state, model->hidden * 4) ||
            !plamo_metal_ensure_buffer(m, &m->normalized, model->hidden * 4) ||
            (m->draft_quantized &&
             !plamo_metal_ensure_buffer(m, &m->draft_normalized, model->hidden * 2)) ||
            !plamo_metal_ensure_buffer(m, &m->branch, model->hidden * 4) ||
            !plamo_metal_ensure_buffer(m, &m->gate, gate * 4) ||
            (m->draft_quantized && !plamo_metal_ensure_buffer(m, &m->draft_gate, gate)) ||
            !plamo_metal_ensure_buffer(m, &m->qkv, qkv * 4) ||
            !plamo_metal_ensure_buffer(m, &m->attention, attention * 4) ||
            !plamo_metal_ensure_buffer(m, &m->logits, model->vocab * 4) ||
            !plamo_metal_ensure_buffer(m, &m->scores, 128 * heads * 4)) {
            plamo_metal_release_decode(m);
            return 0;
        }
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct decode_layer *d = &m->layers[i];
            const uint16_t *weights[] = {l->pre,       l->post,           l->mlp->pre,
                                         l->mlp->post, l->mixer->qkv->qn, l->mixer->qkv->kn};
            const NSUInteger offsets[] = {d->pre, d->post, d->mlp_pre, d->mlp_post, d->qn, d->kn};
            for (size_t j = 0; j < 6; ++j)
                memcpy((char *)m->decode_norms.contents + offsets[j], weights[j],
                       (size_t)((j < 4 ? model->hidden : l->mixer->qkv->dim) * 2));
        }
        memcpy((char *)m->decode_norms.contents + m->final_norm, model->norm,
               (size_t)(model->hidden * 2));
        if (!prepare_verification_scratch(m, model)) {
            plamo_metal_release_decode(m);
            return 0;
        }
        m->model = model;
        return 1;
    }
}

static void encode_norm(struct plamo_metal *m, id<MTLComputeCommandEncoder> e, NSUInteger offset,
                        uint64_t hidden, uint64_t tokens, NSUInteger state_offset) {
    uint32_t p[] = {(uint32_t)hidden, 0, (uint32_t)tokens};
    int auxiliary = m->draft_quantized && tokens == 1;
    [e setComputePipelineState:auxiliary ? m->draft_norm : m->norm];
    [e setBuffer:m->state offset:state_offset atIndex:0];
    [e setBuffer:m->decode_norms offset:offset atIndex:1];
    [e setBuffer:m->normalized offset:0 atIndex:2];
    [e setBytes:p length:sizeof p atIndex:3];
    if (auxiliary)
        [e setBuffer:m->draft_normalized offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

static void encode_residual(struct plamo_metal *m, id<MTLComputeCommandEncoder> e,
                            NSUInteger offset, uint64_t hidden, uint64_t tokens, int mlp) {
    uint32_t p[] = {(uint32_t)hidden, 0, (uint32_t)tokens};
    [e setComputePipelineState:mlp ? m->residual : m->decode_residual];
    [e setBuffer:m->state offset:0 atIndex:0];
    [e setBuffer:m->decode_norms offset:offset atIndex:1];
    [e setBuffer:m->branch offset:0 atIndex:2];
    [e setBytes:p length:sizeof p atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

/* Resolve all four layer projections before encoding; a miss is atomic. */
static int resolve_layer(struct plamo_metal *m, const struct layer *l, struct decode_layer *d) {
    const struct qkv *q = l->mixer->qkv;
    uint64_t h = l->hidden;
    return plamo_metal_resolve_projection(m, q->weight, (q->qh + 2 * q->kh) * q->dim, h, &d->qkv,
                                          1) &&
           plamo_metal_resolve_projection(m, l->mixer->weight, h, q->qh * q->dim, &d->attention,
                                          1) &&
           plamo_metal_resolve_projection(m, l->mlp->gu, l->mlp->intermediate * 2, h, &d->gate,
                                          1) &&
           plamo_metal_resolve_projection(m, l->mlp->down, h, l->mlp->intermediate, &d->down, 1);
}

static int compact_mlp_input(struct plamo_metal *m, uint64_t tokens, uint64_t intermediate) {
    // Compact the input once the strided gate/up working set exceeds 32 MiB.
    return m->swiglu_compact && intermediate && tokens > 4 * MIB / intermediate;
}

/* Decode and prefill use the same FP32 MLP and the same dispatch sequence. */
static void encode_layer_mlp(struct plamo_metal *m, id<MTLComputeCommandEncoder> e,
                             const struct mlp *mlp, const struct decode_layer *d, uint64_t h,
                             uint64_t tokens) {
    encode_norm(m, e, d->mlp_pre, h, tokens, 0);
    plamo_metal_encode_projection(m, e, &d->gate, m->normalized, h, m->gate, 2 * mlp->intermediate,
                                  h, tokens);
    uint32_t params[] = {(uint32_t)h, (uint32_t)mlp->intermediate, (uint32_t)tokens};
    int auxiliary = m->draft_quantized && tokens == 1;
    int compact = !auxiliary && compact_mlp_input(m, tokens, mlp->intermediate);
    [e setComputePipelineState:auxiliary ? m->draft_swiglu
                               : compact ? m->swiglu_compact
                                         : m->swiglu];
    [e setBuffer:m->gate offset:0 atIndex:0];
    [e setBytes:params length:sizeof params atIndex:1];
    if (auxiliary)
        [e setBuffer:m->draft_gate offset:0 atIndex:2];
    else if (compact)
        [e setBuffer:m->normalized offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((mlp->intermediate + 127) / 128, tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
    plamo_metal_encode_projection(m, e, &d->down, compact ? m->normalized : m->gate,
                                  (compact ? 1 : 2) * mlp->intermediate, m->branch, h,
                                  mlp->intermediate, tokens);
    encode_residual(m, e, d->mlp_post, h, tokens, 1);
}

int plamo_metal_forward(struct plamo_metal *m, struct model *model, uint64_t token) {
    m->verification_tokens = 0;
    if (token >= model->vocab)
        return -1;
    if (m->config.memory == PLAMO_METAL_STREAM)
        return 0;
    @autoreleasepool {
        plamo_metal_maintain(m);
        if (!plamo_metal_prepare_model(m, model))
            return -1;
        uint64_t h = model->hidden, scores = 0;
        // Preflight the whole graph before touching embedding, KV or state.
        // No weight acquisition/eviction takes place during command encoding.
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            struct decode_layer *d = &m->layers[i];
            if (c->count > c->capacity || (c->count == c->capacity && !c->sliding))
                return -1;
            if (!resolve_layer(m, l, d))
                return 0;
            uint64_t n = c->count < c->capacity ? c->count + 1 : c->capacity;
            if (n * q->qh > scores)
                scores = n * q->qh;
            if (!prepare_rope(m, &m->ropes[d->rope], n) ||
                !cache_view(m, &d->k, c->k, n, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, n, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, l->mixer->rotation->out, n, c->width, c->capacity))
                return -1;
        }
        if (!plamo_metal_resolve_projection(m, model->head, model->vocab, h, &m->head, 1))
            return 0;
        if (!prepare_graph(m))
            return 0;
        if (!plamo_metal_ensure_buffer(m, &m->scores, scores * 4))
            return -1;
        // A previous batched prefill can have resized these reusable buffers;
        // using a prefix is safe. Embedding expansion preserves every BF16 bit.
        uint32_t *state = m->state.contents;
        for (uint64_t i = 0; i < h; ++i)
            state[i] = (uint32_t)model->embedding[token * h + i] << 16;
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
        if (!command || !e) {
            plamo_metal_problem(m, "cannot create complete Metal decode command");
            return -1;
        }
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            const struct mlp *mlp = l->mlp;
            const struct decode_layer *d = &m->layers[i];
            id<MTLBuffer> table = m->ropes[d->rope].buffer;
            uint32_t p[] = {(uint32_t)q->qh, (uint32_t)q->kh, (uint32_t)q->dim,
                            (uint32_t)(c->count < c->capacity ? c->count + 1 : c->capacity)};
            uint32_t shift = c->count == c->capacity;
            encode_norm(m, e, d->pre, h, 1, 0);
            plamo_metal_encode_projection(m, e, &d->qkv, m->normalized, h, m->qkv,
                                          (q->qh + 2 * q->kh) * q->dim, h, 1);
            if (shift || r->processed_count != c->count ||
                r->scale_bits != m->ropes[d->rope].scale) {
                [e setComputePipelineState:m->decode_keys];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:0];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:1];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:2];
                [e setBuffer:table offset:0 atIndex:3];
                [e setBytes:p length:sizeof p atIndex:4];
                [e setBytes:&shift length:sizeof shift atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            [e setComputePipelineState:m->decode_qkv];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:m->decode_norms offset:d->qn atIndex:1];
            [e setBuffer:m->decode_norms offset:d->kn atIndex:2];
            [e setBuffer:d->k.buffer offset:d->k.offset atIndex:3];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:4];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:5];
            [e setBuffer:table offset:0 atIndex:6];
            [e setBytes:p length:sizeof p atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(q->qh + q->kh, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e setComputePipelineState:m->decode_attention];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:1];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
            [e setBuffer:m->attention offset:0 atIndex:3];
            [e setBuffer:m->scores offset:0 atIndex:4];
            [e setBytes:p length:sizeof p atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(q->qh, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            plamo_metal_encode_projection(m, e, &d->attention, m->attention, q->qh * q->dim,
                                          m->branch, h, q->qh * q->dim, 1);
            encode_residual(m, e, d->post, h, 1, 0);
            encode_layer_mlp(m, e, mlp, d, h, 1);
            m->info.dispatches += 2;
        }
        encode_norm(m, e, m->final_norm, h, 1, 0);
        plamo_metal_encode_projection(m, e, &m->head, m->normalized, h, m->logits, model->vocab, h,
                                      1);
        if (!plamo_metal_complete(m, command, e))
            return -1;
        memcpy(model->state, m->state.contents, (size_t)(h * 4));
        memcpy(model->normalized, m->normalized.contents, (size_t)(h * 4));
        memcpy(model->logits, m->logits.contents, (size_t)(model->vocab * 4));
        for (uint64_t i = 0; i < model->count; ++i) {
            struct mixer *mix = model->layers[i]->mixer;
            if (mix->cache->count < mix->cache->capacity)
                ++mix->cache->count;
            mix->rotation->processed_count = mix->cache->count;
            mix->rotation->scale_bits = m->ropes[m->layers[i].rope].scale;
        }
        ++m->info.fused_decodes;
        m->info.fused_mlps += model->count;
        return 1;
    }
}

static id<MTLComputeCommandEncoder> final_prompt_row(struct plamo_metal *m,
                                                     id<MTLCommandBuffer> command,
                                                     id<MTLComputeCommandEncoder> encoder,
                                                     uint64_t tokens, uint64_t hidden,
                                                     uint64_t attention) {
    [encoder endEncoding];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!blit) {
        plamo_metal_problem(m, "cannot select final prompt row");
        return nil;
    }
    [blit copyFromBuffer:m->attention
             sourceOffset:(tokens - 1) * attention * 4
                 toBuffer:m->attention
        destinationOffset:0
                     size:attention * 4];
    [blit copyFromBuffer:m->state
             sourceOffset:(tokens - 1) * hidden * 4
                 toBuffer:m->state
        destinationOffset:0
                     size:hidden * 4];
    [blit endEncoding];
    encoder = [command computeCommandEncoder];
    if (!encoder)
        plamo_metal_problem(m, "cannot finish final prompt row");
    return encoder;
}

static int prefill_batch(struct plamo_metal *m, struct model *model, const uint32_t *ids,
                         uint64_t tokens, float *states, int verification) {
    m->verification_tokens = 0;
    if (!model || !ids || !tokens)
        return -1;
    if (m->config.memory == PLAMO_METAL_STREAM || tokens > 4096)
        return 0;
    for (uint64_t t = 0; t < tokens; ++t)
        if (ids[t] >= model->vocab)
            return -1;
    @autoreleasepool {
        plamo_metal_maintain(m);
        if (!plamo_metal_prepare_model(m, model))
            return -1;
        uint64_t h = model->hidden, qkv = 0, attention = 0, gate = 0, score_row = 0;
        // Resolve every retained or file-backed tile before state/KV changes. Maintenance
        // cannot evict a projection while this single command is encoded.
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            struct decode_layer *d = &m->layers[i];
            if (c->count > c->capacity || (!c->sliding && tokens > c->capacity - c->count))
                return -1;
            if (c->count > UINT32_MAX - tokens)
                return 0;
            if (!resolve_layer(m, l, d))
                return 0;
            uint64_t count = c->count + tokens;
            if (count > c->capacity)
                count = c->capacity;
            if (count * q->qh > score_row)
                score_row = count * q->qh;
            if (!prepare_rope(m, &m->ropes[d->rope], count) ||
                !cache_view(m, &d->k, c->k, count, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, count, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, l->mixer->rotation->out, count, c->width, c->capacity))
                return -1;
            uint64_t width = (q->qh + 2 * q->kh) * q->dim;
            if (verification) {
                if (!plamo_metal_ensure_buffer(m, &d->proposal, tokens * width * 4) ||
                    !plamo_metal_ensure_buffer(m, &d->proposal_rotated, tokens * c->width * 4))
                    return -1;
                d->verify_prefix = c->count;
            }
            if (width > qkv)
                qkv = width;
            if (q->qh * q->dim > attention)
                attention = q->qh * q->dim;
            if (2 * l->mlp->intermediate > gate)
                gate = 2 * l->mlp->intermediate;
        }
        if (!plamo_metal_resolve_projection(m, model->head, model->vocab, h, &m->head, 1))
            return 0;
        if (!prepare_graph(m))
            return 0;
        // Cap attention scratch at 16 MiB (or a single score row for a larger
        // native context). Dispatches reuse it in query tiles, without waits.
        if (score_row > m->device.maxBufferLength / 4)
            return 0;
        uint64_t queries = 4 * MIB / score_row;
        if (!queries)
            queries = 1;
        if (queries > tokens)
            queries = tokens;
        uint64_t normalized = h;
        if (compact_mlp_input(m, tokens, gate / 2) && gate / 2 > normalized)
            normalized = gate / 2;
        if (!plamo_metal_ensure_buffer(m, &m->state, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->normalized, tokens * normalized * 4) ||
            !plamo_metal_ensure_buffer(m, &m->branch, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->gate, tokens * gate * 4) ||
            !plamo_metal_ensure_buffer(m, &m->qkv, tokens * qkv * 4) ||
            !plamo_metal_ensure_buffer(m, &m->attention, tokens * attention * 4) ||
            !plamo_metal_ensure_buffer(m, &m->scores, queries * score_row * 4) ||
            !plamo_metal_ensure_buffer(m, &m->logits,
                                       (verification ? tokens : 1) * model->vocab * 4))
            return -1;
        uint32_t *state = m->state.contents;
        // Independent rows may fault on different file pages. Read bounded
        // batches concurrently without retaining a second embedding table.
        dispatch_apply((tokens + 15) / 16, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t batch) {
                         uint64_t end = (batch + 1) * 16;
                         if (end > tokens)
                             end = tokens;
                         for (uint64_t t = batch * 16; t < end; ++t)
                             for (uint64_t col = 0; col < h; ++col)
                                 state[t * h + col] =
                                     (uint32_t)model->embedding[(uint64_t)ids[t] * h + col] << 16;
                       });
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
        if (!command || !e) {
            plamo_metal_problem(m, "cannot create complete Metal prefill command");
            return -1;
        }
        uint64_t final_state = tokens - 1;
        m->verify_encoding = verification;
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            const struct mlp *mlp = l->mlp;
            const struct decode_layer *d = &m->layers[i];
            id<MTLBuffer> table = m->ropes[d->rope].buffer;
            id<MTLBuffer> qkv_buffer = verification ? d->proposal : m->qkv;
            uint32_t p[] = {(uint32_t)q->qh,
                            (uint32_t)q->kh,
                            (uint32_t)q->dim,
                            (uint32_t)c->count,
                            (uint32_t)tokens,
                            (uint32_t)c->capacity,
                            0,
                            (uint32_t)verification};
            encode_norm(m, e, d->pre, h, tokens, 0);
            plamo_metal_encode_projection(m, e, &d->qkv, m->normalized, h, qkv_buffer,
                                          (q->qh + 2 * q->kh) * q->dim, h, tokens);
            if (c->count && (verification || c->count + tokens <= c->capacity) &&
                (r->processed_count != c->count || r->scale_bits != m->ropes[d->rope].scale)) {
                uint32_t dp[] = {p[0], p[1], p[2], p[3] + 1}, shift = 0;
                [e setComputePipelineState:m->decode_keys];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:0];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:1];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:2];
                [e setBuffer:table offset:0 atIndex:3];
                [e setBytes:dp length:sizeof dp atIndex:4];
                [e setBytes:&shift length:sizeof shift atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            int tiled_qkv = q->dim == 128 && tokens >= 32 && m->prefill_qkv8;
            [e setComputePipelineState:tiled_qkv ? m->prefill_qkv8 : m->prefill_qkv];
            [e setBuffer:qkv_buffer offset:0 atIndex:0];
            [e setBuffer:m->decode_norms offset:d->qn atIndex:1];
            [e setBuffer:m->decode_norms offset:d->kn atIndex:2];
            [e setBuffer:d->k.buffer offset:d->k.offset atIndex:3];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:4];
            [e setBuffer:verification ? d->proposal_rotated : d->rotated.buffer
                  offset:verification ? 0 : d->rotated.offset
                 atIndex:5];
            [e setBuffer:table offset:0 atIndex:6];
            [e setBytes:p length:sizeof p atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(tiled_qkv ? (q->qh + q->kh + 7) / 8 : q->qh + q->kh,
                                                tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            unsigned attention_tile = c->count + tokens > 128;
            if (attention_tile && m->prompt_matrix_ready) {
                unsigned matrix_tile = c->count + tokens <= 256 ? 2 : 3;
                if (m->prompt_attention[matrix_tile])
                    attention_tile = matrix_tile;
            }
            if (!verification && q->dim == 128 && c->count + tokens <= c->capacity &&
                c->count + tokens <= 512 && m->prompt_attention[attention_tile]) {
                [e setComputePipelineState:m->prompt_attention[attention_tile]];
                [e setBuffer:qkv_buffer offset:0 atIndex:0];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:1];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
                [e setBuffer:m->attention offset:0 atIndex:3];
                [e setBytes:p length:sizeof p atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(q->qh, (tokens + 7) / 8, 1)
                    threadsPerThreadgroup:MTLSizeMake(attention_tile >= 2 ? 256 : 128, 1, 1)];
                ++m->info.dispatches;
            } else if (verification && q->dim == 128 && c->count + tokens <= c->capacity &&
                       queries == tokens) {
                [e setComputePipelineState:m->verify_scores];
                [e setBuffer:qkv_buffer offset:0 atIndex:0];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:1];
                [e setBuffer:d->proposal_rotated offset:0 atIndex:2];
                [e setBuffer:m->scores offset:0 atIndex:3];
                [e setBytes:p length:sizeof p atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(q->qh, (tokens + 3) / 4,
                                                    (c->count + tokens + 15) / 16)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [e setComputePipelineState:m->verify_softmax];
                [e setBuffer:m->scores offset:0 atIndex:0];
                [e setBytes:p length:sizeof p atIndex:1];
                [e dispatchThreadgroups:MTLSizeMake(q->qh, tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e setComputePipelineState:m->verify_values];
                [e setBuffer:qkv_buffer offset:0 atIndex:0];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:1];
                [e setBuffer:m->scores offset:0 atIndex:2];
                [e setBuffer:m->attention offset:0 atIndex:3];
                [e setBytes:p length:sizeof p atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(q->qh, (tokens + 1) / 2, 2)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                m->info.dispatches += 3;
            } else {
                [e setComputePipelineState:m->prefill_attention];
                [e setBuffer:qkv_buffer offset:0 atIndex:0];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:1];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:3];
                [e setBuffer:table offset:0 atIndex:4];
                [e setBuffer:m->attention offset:0 atIndex:5];
                [e setBuffer:m->scores offset:0 atIndex:6];
                [e setBuffer:verification ? d->proposal_rotated : d->rotated.buffer
                      offset:verification ? 0 : d->rotated.offset
                     atIndex:8];
                for (uint64_t first = 0; first < tokens; first += queries) {
                    uint64_t count = tokens - first;
                    if (count > queries)
                        count = queries;
                    p[6] = (uint32_t)first;
                    [e setBytes:p length:sizeof p atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake(q->qh, count, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    ++m->info.dispatches;
                }
            }
            if (!verification && c->count + tokens > c->capacity) {
                [e setComputePipelineState:m->prefill_store];
                [e setBuffer:qkv_buffer offset:0 atIndex:0];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:1];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:3];
                [e setBuffer:table offset:0 atIndex:4];
                [e setBytes:p length:sizeof p atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            uint64_t outputs = tokens;
            if (!states && !verification && tokens > 1 && i + 1 == model->count) {
                // Every prompt K/V is ready. Only the final hidden state can
                // affect generation, so the last projection/MLP needs one row.
                e = final_prompt_row(m, command, e, tokens, h, q->qh * q->dim);
                if (!e)
                    return -1;
                outputs = 1;
                final_state = 0;
            }
            plamo_metal_encode_projection(m, e, &d->attention, m->attention, q->qh * q->dim,
                                          m->branch, h, q->qh * q->dim, outputs);
            encode_residual(m, e, d->post, h, outputs, 0);
            encode_layer_mlp(m, e, mlp, d, h, outputs);
        }
        encode_norm(m, e, m->final_norm, h, verification ? tokens : 1,
                    verification ? 0 : final_state * h * 4);
        plamo_metal_encode_projection(m, e, &m->head, m->normalized, h, m->logits, model->vocab, h,
                                      verification ? tokens : 1);
        int completed = plamo_metal_complete(m, command, e);
        m->verify_encoding = 0;
        if (!completed)
            return -1;
        if (verification) {
            m->verification_tokens = tokens;
            m->verification_index = 0;
            ++m->info.verification_batches;
            m->info.verified_tokens += tokens;
            return 1;
        }
        if (states)
            memcpy(states, m->state.contents, (size_t)(tokens * h * 4));
        memcpy(model->state, (float *)m->state.contents + final_state * h, (size_t)(h * 4));
        memcpy(model->normalized, m->normalized.contents, (size_t)(h * 4));
        memcpy(model->logits, m->logits.contents, (size_t)(model->vocab * 4));
        for (uint64_t i = 0; i < model->count; ++i) {
            struct mixer *mix = model->layers[i]->mixer;
            mix->cache->count += tokens;
            if (mix->cache->count > mix->cache->capacity)
                mix->cache->count = mix->cache->capacity;
            mix->rotation->processed_count = mix->cache->count;
            mix->rotation->scale_bits = m->ropes[m->layers[i].rope].scale;
        }
        ++m->info.fused_prefills;
        m->info.fused_mlps += model->count;
        return 1;
    }
}

int plamo_metal_prefill(struct plamo_metal *m, struct model *model, const uint32_t *ids,
                        uint64_t tokens, float *states) {
    return prefill_batch(m, model, ids, tokens, states, 0);
}

int plamo_metal_verify(struct plamo_metal *m, struct model *model, const uint32_t *ids,
                       uint64_t tokens, uint32_t *next) {
    if (!next || !tokens || tokens > PLAMO_VERIFY_MAX)
        return 0;
    int result = prefill_batch(m, model, ids, tokens, NULL, 1);
    if (result == 1) {
        const float *logits = m->logits.contents;
        for (uint64_t i = 0; i < tokens; ++i)
            next[i] = (uint32_t)plamo_argmax(logits + i * model->vocab, model->vocab);
    }
    return result;
}

int plamo_metal_commit(struct plamo_metal *m, struct model *model, uint64_t index) {
    if (m->model != model || index != m->verification_index || index >= m->verification_tokens)
        return 0;
    int full = 0;
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct cache *c = model->layers[i]->mixer->cache;
        uint64_t count = m->layers[i].verify_prefix + index;
        if (count > c->capacity)
            count = c->capacity;
        if (c->count != count)
            return 0;
        if (c->count == c->capacity) {
            if (!c->sliding)
                return 0;
            full = 1;
        }
    }
    @autoreleasepool {
        // Rebase full sliding windows on the GPU. Ordinary appends only copy
        // values and precomputed rotations; they introduce no new arithmetic.
        if (full) {
            id<MTLCommandBuffer> command = [m->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
            if (!command || !e) {
                plamo_metal_problem(m, "cannot commit verified KV");
                return -1;
            }
            for (uint64_t i = 0; i < model->count; ++i) {
                const struct mixer *mix = model->layers[i]->mixer;
                const struct qkv *q = mix->qkv;
                const struct cache *c = mix->cache;
                const struct decode_layer *d = &m->layers[i];
                if (c->count != c->capacity)
                    continue;
                uint32_t p[] = {(uint32_t)q->qh,
                                (uint32_t)q->kh,
                                (uint32_t)q->dim,
                                (uint32_t)c->count,
                                1,
                                (uint32_t)c->capacity,
                                0,
                                0};
                uint64_t width = (q->qh + 2 * q->kh) * q->dim;
                [e setComputePipelineState:m->prefill_store];
                [e setBuffer:d->proposal offset:(NSUInteger)(index * width * 4) atIndex:0];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:1];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:3];
                [e setBuffer:m->ropes[d->rope].buffer offset:0 atIndex:4];
                [e setBytes:p length:sizeof p atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            if (!plamo_metal_complete(m, command, e))
                return -1;
        }
        for (uint64_t i = 0; i < model->count; ++i) {
            struct mixer *mix = model->layers[i]->mixer;
            const struct qkv *q = mix->qkv;
            struct cache *c = mix->cache;
            const struct decode_layer *d = &m->layers[i];
            if (c->count < c->capacity) {
                uint64_t width = (q->qh + 2 * q->kh) * q->dim;
                const float *values = (const float *)d->proposal.contents + index * width;
                memcpy(c->k + c->count * c->width, values + q->qh * q->dim, (size_t)c->width * 4);
                memcpy(c->v + c->count * c->width, values + (q->qh + q->kh) * q->dim,
                       (size_t)c->width * 4);
                memcpy(mix->rotation->out + c->count * c->width,
                       (const float *)d->proposal_rotated.contents + index * c->width,
                       (size_t)c->width * 4);
                ++c->count;
            }
            mix->rotation->processed_count = c->count;
            mix->rotation->scale_bits = m->ropes[d->rope].scale;
        }
        memcpy(model->state, (const float *)m->state.contents + index * model->hidden,
               (size_t)model->hidden * 4);
        memcpy(model->normalized, (const float *)m->normalized.contents + index * model->hidden,
               (size_t)model->hidden * 4);
        memcpy(model->logits, (const float *)m->logits.contents + index * model->vocab,
               (size_t)model->vocab * 4);
        ++m->verification_index;
        ++m->info.committed_tokens;
        return 1;
    }
}
