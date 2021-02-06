/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "KVStoreInternalClient.cuh"

#ifndef KVGPU_KVSTORECTX_CUH
#define KVGPU_KVSTORECTX_CUH


template<typename K, typename V, typename M>
class KVStoreCtx {
public:
    KVStoreCtx() : k() {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig>& conf) : k(conf) {

    }


    ~KVStoreCtx() {}

    std::unique_ptr<KVStoreInternalClient<K, V, M>> getClient() {
        return std::make_unique<KVStoreInternalClient<K, V, M>>(k.getSlab(), k.getCache(), k.getModel());
    }

private:
    KVStore<K, V, M> k;
};

template<typename K, typename M>
class KVStoreCtx<K,data_t, M> {
public:
    KVStoreCtx() : k() {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig>& conf) : k(conf) {

    }


    ~KVStoreCtx() {}

    std::unique_ptr<KVStoreInternalClient<K, data_t, M>> getClient() {
        return std::make_unique<KVStoreInternalClient<K, data_t, M>>(k.getSlab(), k.getCache(), k.getModel());
    }

private:
    KVStore<K, data_t*, M> k;
};


template<typename K, typename V, typename M>
class NoCacheKVStoreCtx {
public:
    NoCacheKVStoreCtx() : k() {

    }

    NoCacheKVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig>& conf) : k(conf) {

    }

    ~NoCacheKVStoreCtx() {}

    std::unique_ptr<NoCacheKVStoreInternalClient<K, V, M>> getClient() {
        return std::make_unique<NoCacheKVStoreInternalClient<K, V, M>>(k.getSlab(), k.getCache(), k.getModel());
    }

private:
    KVStore<K, V, M> k;
};

template<typename K, typename V, typename M>
class JustCacheKVStoreCtx {
public:
    JustCacheKVStoreCtx() : k() {

    }

    JustCacheKVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig>& conf) : k(conf) {

    }

    ~JustCacheKVStoreCtx() {}

    std::unique_ptr<JustCacheKVStoreInternalClient<K, V, M>> getClient() {
        return std::make_unique<JustCacheKVStoreInternalClient<K, V, M>>(k.getSlab(), k.getCache(), k.getModel());
    }

private:
    KVStore<K, V, M> k;
};


#endif //KVGPU_KVSTORECTX_CUH
