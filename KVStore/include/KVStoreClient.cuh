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

#include <utility>
#include "KVStoreInternalClient.cuh"
#include <memory>
#include <atomic>
#include <exception>
#include "KVStoreCtx.cuh"

#ifndef KVGPU_KVSTORECLIENT_CUH
#define KVGPU_KVSTORECLIENT_CUH

template<typename K, typename V, typename M>
class KVStoreClient {
public:

    KVStoreClient() = delete;

    explicit KVStoreClient(KVStoreCtx<K, V, M> ctx) : client(std::move(ctx.getClient())) {

    }

    KVStoreClient(const KVStoreClient<K, V, M> &) = delete;

    KVStoreClient(KVStoreClient<K, V, M> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~KVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf, std::vector<std::chrono::high_resolution_clock::time_point>& times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

    std::future<void> change_model(M &newModel, block_t *block, double& time) {
        return client->change_model(newModel, block, time);
    }

    M getModel(){
        return client->getModel();
    }

private:
    std::unique_ptr<KVStoreInternalClient<K, V, M>> client;
};

template<typename K, typename M>
class KVStoreClient<K, data_t, M> {
public:

    KVStoreClient() = delete;

    explicit KVStoreClient(KVStoreCtx<K, data_t, M> ctx) : client(std::move(ctx.getClient())) {

    }

    KVStoreClient(const KVStoreClient<K, data_t, M> &) = delete;

    KVStoreClient(KVStoreClient<K, data_t, M> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~KVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, data_t*>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>>& resBuf, std::vector<std::chrono::high_resolution_clock::time_point>& times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

    std::future<void> change_model(M &newModel, block_t *block, double& time) {
        return client->change_model(newModel, block, time);
    }

    M getModel(){
        return client->getModel();
    }

private:
    std::unique_ptr<KVStoreInternalClient<K, data_t, M>> client;
};


template<typename K, typename V, typename M>
class NoCacheKVStoreClient {
public:

    NoCacheKVStoreClient() = delete;

    explicit NoCacheKVStoreClient(NoCacheKVStoreCtx<K, V, M> ctx) : client(std::move(ctx.getClient())) {

    }

    NoCacheKVStoreClient(const NoCacheKVStoreClient<K, V, M> &) = delete;

    NoCacheKVStoreClient(NoCacheKVStoreClient<K, V, M> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~NoCacheKVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf, std::vector<std::chrono::high_resolution_clock::time_point>& times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

private:
    std::unique_ptr<NoCacheKVStoreInternalClient<K, V, M>> client;
};

template<typename K, typename V, typename M>
class JustCacheKVStoreClient {
public:

    JustCacheKVStoreClient() = delete;

    explicit JustCacheKVStoreClient(JustCacheKVStoreCtx<K, V, M> ctx) : client(std::move(ctx.getClient())) {

    }

    JustCacheKVStoreClient(const JustCacheKVStoreClient<K, V, M> &) = delete;

    JustCacheKVStoreClient(JustCacheKVStoreClient<K, V, M> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~JustCacheKVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf, std::vector<std::chrono::high_resolution_clock::time_point>& times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

private:
    std::unique_ptr<JustCacheKVStoreInternalClient<K, V, M>> client;
};

#endif //KVGPU_KVSTORECLIENT_CUH
