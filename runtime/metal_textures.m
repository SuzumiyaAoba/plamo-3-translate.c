#include "metal_internal.h"
#include <string.h>

// Texture compression is lossless. Move the sign into the low bit so nearby
// exponents remain nearby unsigned colors; shader reads reverse this rotation
// before any floating-point operation. The cache retains only one weight copy.
id<MTLTexture> plamo_metal_texture(struct plamo_metal *m, const void *source, id<MTLBuffer> buffer,
                                  uint64_t bytes, uint64_t remaining, uint64_t *allocated) {
    if (!m->texture_weights || (bytes & 1) || !bytes) return nil;
    uint64_t width = 1024;
    for (const struct plamo_weight_spec *s = m->config.schema; s && s->name; ++s) {
        if (s->rank != 2 || !s->slot || !s->shape || !s->shape[0] || !s->shape[1] ||
            s->shape[1] > UINT64_MAX / 2 || s->shape[0] > UINT64_MAX / (s->shape[1] * 2)) continue;
        uintptr_t base = (uintptr_t)*s->slot, address = (uintptr_t)source;
        uint64_t tensor_bytes = s->shape[0] * s->shape[1] * 2;
        if (address < base || address - base >= tensor_bytes || bytes > tensor_bytes - (address - base)) continue;
        uint64_t pixels = s->shape[1] / 4;
        if (!(s->shape[1] % 4) && pixels >= 32 && !(pixels & (pixels - 1)) && pixels <= 16384) width = pixels;
        break;
    }
    uint64_t row_bytes = width * 8, height = (bytes + row_bytes - 1) / row_bytes;
    // The CPU read buffer also holds the final texture row's zero padding.
    if (!height || height > 16384 || height > buffer.length / row_bytes) return nil;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Uint
                                                                               width:width height:height mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead;
    d.allowGPUOptimizedContents = YES;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 120500
    if (@available(macOS 12.5, *)) d.compressionType = MTLTextureCompressionTypeLossless;
#endif
    // Earlier SDKs also default to lossless texture optimization.
    MTLSizeAndAlign estimate = [m->device heapTextureSizeAndAlignWithDescriptor:d];
    if (!estimate.size || estimate.size > remaining) return nil;
    // Allocate only after the fully resident proposer has reserved its space.
    // The heap owns no activations and never aliases live weight resources.
    if (m->texture_heap_budget && !m->texture_heap_attempted) {
        m->texture_heap_attempted = 1;
        // A single oversized heap can fail despite a sufficient total budget.
        // Include the first image's layout overhead and bound the allocation;
        // any later images that do not fit retain the individual-texture path.
        uint64_t size = m->texture_heap_budget, minimum = plamo_metal_aligned(estimate.size);
        if (size < minimum) size = minimum;
        if (size > m->device.maxBufferLength) size = m->device.maxBufferLength / 16384 * 16384;
        if (size > remaining) size = remaining / 16384 * 16384;
        MTLHeapDescriptor *heap = [[MTLHeapDescriptor alloc] init];
        heap.size = size;
        heap.storageMode = MTLStorageModePrivate;
        heap.hazardTrackingMode = MTLHazardTrackingModeTracked;
        if (size >= estimate.size) m->texture_heap = [m->device newHeapWithDescriptor:heap];
        [heap release];
        if (m->texture_heap && m->texture_heap.size > remaining) {
            [m->texture_heap release]; m->texture_heap = nil;
        }
    }
    id<MTLTexture> texture = m->texture_heap ? [m->texture_heap newTextureWithDescriptor:d] : nil;
    if (!texture && m->texture_heap && !m->texture_heap.usedSize) {
        [m->texture_heap release]; m->texture_heap = nil;
    }
    if (!texture) texture = [m->device newTextureWithDescriptor:d];
    if (!texture) return nil;
    if (!texture.allocatedSize || texture.allocatedSize > remaining) { [texture release]; return nil; }
    uint16_t *values = buffer.contents;
    for (uint64_t i = 0; i < bytes / 2; ++i) values[i] = (uint16_t)((values[i] << 1) | (values[i] >> 15));
    memset((char *)values + bytes, 0, (size_t)(height * row_bytes - bytes));
    plamo_metal_residency_add(m, texture);
    id<MTLCommandBuffer> command = [m->queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    if (command && encoder) {
        [encoder copyFromBuffer:buffer sourceOffset:0 sourceBytesPerRow:row_bytes sourceBytesPerImage:height * row_bytes
                    sourceSize:MTLSizeMake(width, height, 1) toTexture:texture destinationSlice:0 destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
        [encoder optimizeContentsForGPUAccess:texture];
        [encoder endEncoding];
        if (plamo_metal_complete(m, command, nil)) {
            *allocated = texture.allocatedSize;
            return texture;
        }
    } else plamo_metal_problem(m, "cannot prepare lossless BF16 texture");
    // A failed conversion leaves the caller's original BF16 buffer usable.
    for (uint64_t i = 0; i < bytes / 2; ++i) values[i] = (uint16_t)((values[i] >> 1) | (values[i] << 15));
    plamo_metal_residency_remove(m, texture); plamo_metal_residency_commit(m);
    [texture release]; return nil;
}
