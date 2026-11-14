#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../runtime/packing.h"
#include "metal_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(ok)                                                                                  \
    do {                                                                                           \
        if (!(ok)) {                                                                               \
            fprintf(stderr, "packing check failed at %d: %s\n", __LINE__, #ok);                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static void complete(id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
    CHECK(command.status == MTLCommandBufferStatusCompleted);
}

static void texture_round_trip(id<MTLDevice> device, id<MTLCommandQueue> queue,
                               id<MTLComputePipelineState> pipeline) {
    enum { WIDTH = 512, HEIGHT = 32, COUNT = WIDTH * HEIGHT * 4 };
    uint16_t *original = malloc(COUNT * 2), *encoded = malloc(COUNT * 2);
    CHECK(original && encoded);
    for (unsigned i = 0; i < COUNT; ++i) {
        uint16_t bits = original[i] = (uint16_t)(i * 40503u + 17u);
        encoded[i] = (uint16_t)((bits << 1) | (bits >> 15));
    }
    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Uint
                                                           width:WIDTH
                                                          height:HEIGHT
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> transfer = [device newTextureWithDescriptor:descriptor];
    CHECK(transfer);
    [transfer replaceRegion:MTLRegionMake2D(0, 0, WIDTH, HEIGHT)
                mipmapLevel:0
                  withBytes:encoded
                bytesPerRow:WIDTH * 8];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.allowGPUOptimizedContents = YES;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    CHECK(texture);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:transfer
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(WIDTH, HEIGHT, 1)
                toTexture:texture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit optimizeContentsForGPUAccess:texture];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    [transfer release];
    id<MTLBuffer> result = [device newBufferWithLength:COUNT * 2
                                               options:MTLResourceStorageModeShared];
    CHECK(result);
    for (uint32_t mode = 0; mode < 4; ++mode) {
        uint32_t offset = mode == 2 ? 4 : mode == 3 ? 1 : 0, count = (COUNT - offset) & ~3u;
        uint32_t params[] = {0, mode < 2 ? WIDTH * 4 : 132, 0, 0, 0, !(offset & 3), offset};
        memset(result.contents, 0xa5, COUNT * 2);
        command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:result offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof count atIndex:2];
        [encoder setBytes:params length:sizeof params atIndex:3];
        [encoder setBytes:&mode length:sizeof mode atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((count + 511) / 512, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        complete(command, encoder);
        CHECK(!memcmp(original + offset, result.contents, count * 2));
        for (uint32_t i = count; i < COUNT; ++i)
            CHECK(((const uint16_t *)result.contents)[i] == 0xa5a5);
    }
    [result release];
    [texture release];
    free(original);
    free(encoded);
    puts(
        "Integer texture BF16: PASS (all 65536 bit patterns, optimized private texture, direct/linear rows, aligned/unaligned aliases, exact round trips)");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int required = argc > 1 && !strcmp(argv[1], "--require-gpu");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyApple7]) {
            fprintf(stderr, "Lossless BF16 tests %s: no Apple Silicon GPU\n",
                    required ? "FAILED" : "SKIPPED");
            [device release];
            return required ? 1 : 0;
        }
        NSString *source = [[NSString stringWithUTF8String:plamo_metal_source]
            stringByAppendingString:
                @"\nkernel void unpack_test(device const ushort *w [[buffer(0)]], device ushort *out [[buffer(1)]], constant uint &count [[buffer(2)]], constant Linear &p [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < count) out[i] = weight_bits<true>(w, i, p); }\n"
                 "kernel void unpack4_test(device const ushort *w [[buffer(0)]], device ushort *out [[buffer(1)]], constant uint &count [[buffer(2)]], constant Linear &p [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i * 4 + 3 < count) { uint4 bits = as_type<uint4>(weight4<true>(w, ulong(i) * 4, p)); *((device ushort4 *)(out + i * 4)) = ushort4(bits >> 16); } }\n"
                 "kernel void texture_bits_test(texture2d<ushort, access::read> w [[texture(0)]], device ushort4 *out [[buffer(1)]], constant uint &count [[buffer(2)]], constant Linear &p [[buffer(3)]], constant uint &mode [[buffer(4)]], uint i [[thread_position_in_grid]]) { if (i * 4 + 3 < count) { uint row = i * 4 / p.columns, col = i * 4 % p.columns; out[i] = mode == 0 ? weight_row_bits4<4>(w, row, col, p) : mode < 3 ? weight_row_bits4<3>(w, row, col, p) : ushort4(as_type<uint4>(weight4<3>(w, ulong(i) * 4, p)) >> 16); } }\n"];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_3;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library)
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        CHECK(library);
        const char *names[] = {"plamo_pack_analyze", "plamo_pack_encode", "unpack_test",
                               "unpack4_test", "texture_bits_test"};
        id<MTLComputePipelineState> pipelines[5];
        for (unsigned i = 0; i < 5; ++i) {
            id<MTLFunction> fn =
                [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
            pipelines[i] = [device newComputePipelineStateWithFunction:fn error:&error];
            [fn release];
            CHECK(pipelines[i]);
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        CHECK(queue);
        texture_round_trip(device, queue, pipelines[4]);
        const uint32_t counts[] = {1, 511, 512, 513, 4097, 65536 + 17};
        for (unsigned pattern = 0; pattern < 3; ++pattern)
            for (unsigned shape = 0; shape < sizeof counts / sizeof *counts; ++shape) {
                uint32_t count = counts[shape],
                         blocks = (count + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
                id<MTLBuffer> raw = [device newBufferWithLength:count * 2
                                                        options:MTLResourceStorageModeShared];
                id<MTLBuffer> meta =
                    [device newBufferWithLength:blocks * sizeof(struct plamo_bf16_block)
                                        options:MTLResourceStorageModeShared];
                id<MTLBuffer> result = [device newBufferWithLength:count * 2
                                                           options:MTLResourceStorageModeShared];
                CHECK(raw && meta && result);
                uint16_t *original = raw.contents;
                for (uint32_t i = 0; i < count; ++i) {
                    if (!pattern)
                        original[i] =
                            (uint16_t)(i * 40503u + 17u); // Every possible BF16 bit pattern.
                    else {
                        unsigned bits = pattern == 1 ? (i / PLAMO_BF16_BLOCK) % 9 : 0;
                        unsigned base = bits == 8 ? 0 : 80, delta = i & ((1u << bits) - 1);
                        original[i] =
                            (uint16_t)(((i & 1) << 15) | ((base + delta) << 7) | ((i * 13) & 127));
                    }
                }
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipelines[0]];
                [encoder setBuffer:raw offset:0 atIndex:0];
                [encoder setBuffer:meta offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof count atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                complete(command, encoder);
                struct plamo_bf16_block *headers = meta.contents;
                for (uint32_t b = 0; b < blocks; ++b) {
                    unsigned minimum = 255, maximum = 0;
                    for (uint32_t i = b * PLAMO_BF16_BLOCK;
                         i < count && i < (b + 1) * PLAMO_BF16_BLOCK; ++i) {
                        unsigned e = (original[i] >> 7) & 255;
                        if (e < minimum)
                            minimum = e;
                        if (e > maximum)
                            maximum = e;
                    }
                    unsigned bits = 0;
                    for (unsigned span = maximum - minimum; span; span >>= 1)
                        ++bits;
                    CHECK(headers[b].exponent == (minimum | bits << 8));
                }
                uint64_t bytes = plamo_bf16_pack_offsets(headers, count);
                CHECK(bytes);
                id<MTLBuffer> packed = [device newBufferWithLength:(NSUInteger)bytes
                                                           options:MTLResourceStorageModeShared];
                CHECK(packed);
                command = [queue commandBuffer];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipelines[1]];
                [encoder setBuffer:raw offset:0 atIndex:0];
                [encoder setBuffer:meta offset:0 atIndex:1];
                [encoder setBuffer:packed offset:0 atIndex:2];
                [encoder setBytes:&count length:sizeof count atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                complete(command, encoder);
                const uint32_t offsets[] = {0, 1, 7, 511};
                for (unsigned o = 0; o < sizeof offsets / sizeof *offsets; ++o) {
                    if (offsets[o] >= count)
                        continue;
                    uint32_t n = count - offsets[o], params[] = {0, 0, 0, 0, 0, 0, offsets[o]};
                    memset(result.contents, 0xa5, count * 2);
                    command = [queue commandBuffer];
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:pipelines[2]];
                    [encoder setBuffer:packed offset:0 atIndex:0];
                    [encoder setBuffer:result offset:0 atIndex:1];
                    [encoder setBytes:&n length:sizeof n atIndex:2];
                    [encoder setBytes:params length:sizeof params atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    complete(command, encoder);
                    CHECK(!memcmp(original + offsets[o], result.contents, n * 2));
                }
                uint32_t params[7] = {0};
                command = [queue commandBuffer];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipelines[3]];
                [encoder setBuffer:packed offset:0 atIndex:0];
                [encoder setBuffer:result offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof count atIndex:2];
                [encoder setBytes:params length:sizeof params atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((count + 511) / 512, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                complete(command, encoder);
                CHECK(!memcmp(original, result.contents, (count / 4) * 8));
                [raw release];
                [meta release];
                [result release];
                [packed release];
            }
        struct plamo_bf16_block invalid = {0, 9 << 8};
        CHECK(!plamo_bf16_pack_offsets(&invalid, 0) &&
              !plamo_bf16_pack_offsets(&invalid, UINT64_MAX));
        CHECK(!plamo_bf16_pack_offsets(&invalid, 1));
        for (unsigned i = 0; i < 5; ++i)
            [pipelines[i] release];
        [queue release];
        [library release];
        [options release];
        [device release];
        puts(
            "Lossless BF16: PASS (all 65536 bit patterns, NaNs/inf/subnormals/signed zero, exponent widths 0..8, tails, offsets, scalar/vector exact round trips)");
    }
    return 0;
}
