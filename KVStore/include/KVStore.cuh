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
#include <memory>
#include <atomic>
#include <KVCache.cuh>
#include <Slab.cuh>
#include <StandardSlabDefinitions.cuh>
#include <mutex>
#include <iostream>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

#ifndef KVGPU_KVSTORE_CUH
#define KVGPU_KVSTORE_CUH

int SLAB_SIZE = 1000000;

const int MAX_ATTEMPTS = 1;

struct PartitionedSlabUnifiedConfig {
    int size;
    int gpu;
    cudaStream_t stream;
};

const std::vector<PartitionedSlabUnifiedConfig> STANDARD_CONFIG = {{SLAB_SIZE, 0, cudaStreamDefault},
                                                                   {SLAB_SIZE, 1, cudaStreamDefault}};

template<typename K, typename V>
class Cache {
public:
    typedef kvgpu::KVCache<K, V, 1000000, 8> type;
};

template<typename V>
struct ResultsBuffers {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new V[s]), size(s), retryGPU(false) {
        for (int i = 0; i < size; i++)
            requestIDs[i] = -1;
    }

    ResultsBuffers(const ResultsBuffers<V> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile V *resultValues;
    int size;
    bool retryGPU;
};

template<>
struct ResultsBuffers<data_t> {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new volatile data_t *[s]), size(s),
                                     retryGPU(false) {
        for (int i = 0; i < size; i++) {
            requestIDs[i] = -1;
            resultValues[i] = nullptr;
        }
    }

    ResultsBuffers(const ResultsBuffers<data_t> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        for (int i = 0; i < size; i++) {
            if (resultValues[i] && resultValues[i]->data)
                delete[] resultValues[i]->data;
        }
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile data_t **resultValues;
    int size;
    bool retryGPU;
};

template<typename K, typename V>
struct BatchData {
    BatchData(int rbStart, std::shared_ptr<ResultsBuffers<V>> rb, int s) : keys(s), values(s), requests(s), hashes(s),
                                                                           requestID(s),
                                                                           handleInCache(s), resBuf(rb),
                                                                           resBufStart(rbStart), size(s), idx(0),
                                                                           flush(false) {
        for (int i = 0; i < s; i++) {
            handleInCache[i] = false;
        }
    }

    ~BatchData() = default;

    std::vector<K> keys;
    std::vector<V> values;
    std::vector<unsigned> requests;
    std::vector<unsigned> hashes;
    std::vector<int> requestID;
    std::vector<bool> handleInCache;
    std::shared_ptr<ResultsBuffers<V>> resBuf;
    int resBufStart;
    int size;
    int idx;
    bool flush;
};

template<typename K>
struct BatchData<K, data_t> {
    BatchData(int rbStart, std::shared_ptr<ResultsBuffers<data_t>> &rb, int s) : keys(s), values(s), requests(s),
                                                                                 hashes(s), requestID(s),
                                                                                 handleInCache(s), resBuf(rb),
                                                                                 resBufStart(rbStart), size(s), idx(0),
                                                                                 flush(false) {
        for (int i = 0; i < s; i++) {
            handleInCache[i] = false;
        }
    }

    ~BatchData() = default;

    std::vector<K> keys;
    std::vector<data_t *> values;
    std::vector<unsigned> requests;
    std::vector<unsigned> hashes;
    std::vector<int> requestID;
    std::vector<bool> handleInCache;
    std::shared_ptr<ResultsBuffers<data_t>> resBuf;
    int resBufStart;
    int size;
    int idx;
    bool flush;
};

struct StatData {
    std::chrono::high_resolution_clock::time_point timestampEnd;
    std::chrono::high_resolution_clock::time_point timestampWriteBack;
    std::chrono::high_resolution_clock::time_point timestampStartBatch;
    std::chrono::high_resolution_clock::time_point timestampDequeueToBatch;
    float duration;
    int size;
    int timesGoingToCache;
};

template<typename K, typename V, typename M>
struct Slabs {

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, V> *> q_t;

    Slabs(const std::vector<PartitionedSlabUnifiedConfig> &config, std::shared_ptr<typename Cache<K, V>::type> cache,
          std::shared_ptr<M> m) : numslabs(config.size()), slabs(new SlabUnified<K, V>[numslabs]),
                                  gpu_qs(new q_t[numslabs]), done(false),
                                  mops(new tbb::concurrent_vector<StatData>[numslabs]), _cache(cache), ops(0), load(0),
                                  model(m) {
        for (int i = 0; i < config.size(); i++) {
            cudaStream_t *stream = new cudaStream_t();
            *stream = config[i].stream;
            slabs[i] = std::move(SlabUnified<K, V>(config[i].size, config[i].gpu, stream));
        }
        for (int i = 0; i < numslabs; ++i) {
            threads.push_back(std::thread([this](int tid) {
                K *keys = slabs[tid].getBatchKeys();
                V *values = slabs[tid].getBatchValues();
                int *requests = slabs[tid].getBatchRequests();
                unsigned *hashes = slabs[tid].getHashValues();

                BatchData<K, V> *holdonto = nullptr;

                std::vector<std::pair<int, BatchData<K, V> *>> writeBack;
                writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                int index = THREADS_PER_BLOCK * BLOCKS;
                while (!done.load()) {
                    writeBack.clear();
                    for (int i = 0; i < index; i++) {
                        requests[i] = REQUEST_EMPTY;
                    }
                    index = 0;

                    BatchData<K, V> *res;

                    auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                    if (holdonto) {
                        //std::cerr << "Hold onto set " << tid << std::endl;
                        writeBack.push_back({index, holdonto});

                        for (int i = 0; i < holdonto->idx; i++) {
                            keys[index + i] = holdonto->keys[i];
                            values[index + i] = holdonto->values[i];
                            requests[index + i] = holdonto->requests[i];
                            hashes[index + i] = holdonto->hashes[i];
                        }
                        index += holdonto->idx;
                        holdonto = nullptr;
                    }

                    int attempts = 0;

                    while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                        if (this->gpu_qs[tid].try_pop(res)) {
                            load--;
                            //std::cerr << "Got a batch on handler thread " << tid << "\n";
                            if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                holdonto = res;
                                break;
                            }
                            for (int i = 0; i < res->idx; i++) {
                                keys[index + i] = res->keys[i];
                                values[index + i] = res->values[i];
                                requests[index + i] = res->requests[i];
                                hashes[index + i] = res->hashes[i];
                            }
                            writeBack.push_back({index, res});
                            index += res->idx;
                            if (res->flush) {
                                break;
                            }
                        } else {
                            attempts++;
                        }
                    }

                    if (index > 0) {

                        //std::cerr << "Batching " << tid << "\n";

                        auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                        float t;

                        this->slabs[tid].diy_batch(t, ceil(index / 512.0), 512);

                        auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                        int timesGoingToCache = 0;
                        for (auto &wb : writeBack) {

                            int rbLoc = wb.second->resBufStart;

                            for (int i = 0; i < wb.second->idx; ++i) {

                                if (wb.second->handleInCache[i]) {
                                    timesGoingToCache++;
                                    auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                *(this->model));
                                    if (cacheRes.first->valid == 1) {
                                        wb.second->resBuf->resultValues[rbLoc + i] = cacheRes.first->value;
                                    } else {
                                        cacheRes.first->valid = 1;
                                        cacheRes.first->value = values[wb.first + i];
                                        cacheRes.first->deleted = (values[wb.first + i] == EMPTY<V>::value);
                                    }
                                    asm volatile("":: : "memory");

                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                } else {
                                    wb.second->resBuf->resultValues[rbLoc + i] = values[wb.first + i];
                                    asm volatile("":: : "memory");
                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                }
                            }
                            delete wb.second;
                        }

                        mops[tid].push_back(
                                {std::chrono::high_resolution_clock::now(), timestampWriteBack, timestampStartBatch,
                                 timestampWriteToBatch, t, index, timesGoingToCache});

                        ops += index;
                        //std::cerr << "Batched " << tid << "\n";

                    }
                }
            }, i));
        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] slabs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    SlabUnified<K, V> *slabs;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};

template<typename K, typename M>
struct Slabs<K, data_t *, M> {

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, data_t> *> q_t;

    Slabs(const std::vector<PartitionedSlabUnifiedConfig> &config,
          std::shared_ptr<typename Cache<K, data_t *>::type> cache, std::shared_ptr<M> m) : done(false),
                                                                                            mops(new tbb::concurrent_vector<StatData>[config.size()]),
                                                                                            _cache(cache), ops(0),
                                                                                            load(0), model(m) {
        std::unordered_map<int, std::shared_ptr<SlabUnified<K, data_t *>>> gpusToSlab;
        for (int i = 0; i < config.size(); i++) {
            if (gpusToSlab.find(config[i].gpu) == gpusToSlab.end())
                gpusToSlab[config[i].gpu] = std::make_shared<SlabUnified<K, data_t *>>(config[i].size, config[i].gpu);
        }
        gpu_qs = new q_t[gpusToSlab.size()];
        numslabs = gpusToSlab.size();

        for (int i = 0; i < config.size(); i++) {
            //config[i].stream;
            threads.push_back(
                    std::thread([this](int tid, int gpu, std::shared_ptr<SlabUnified<K, data_t *>> slab,
                                       cudaStream_t stream) {
                                    slab->setGPU();
                                    auto batchData = new BatchBuffer<K, data_t *>();

                                    K *keys = batchData->getBatchKeys();
                                    data_t **values = batchData->getBatchValues();
                                    int *requests = batchData->getBatchRequests();
                                    unsigned *hashes = batchData->getHashValues();

                                    BatchData<K, data_t> *holdonto = nullptr;

                                    std::vector<std::pair<int, BatchData<K, data_t> *>> writeBack;
                                    writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                                    int index = THREADS_PER_BLOCK * BLOCKS;
                                    while (!done.load()) {
                                        writeBack.clear();
                                        for (int i = 0; i < index; i++) {
                                            requests[i] = REQUEST_EMPTY;
                                        }
                                        index = 0;

                                        BatchData<K, data_t> *res;

                                        auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                                        if (holdonto) {
                                            //std::cerr << "Hold onto set " << tid << std::endl;
                                            writeBack.push_back({index, holdonto});

                                            for (int i = 0; i < holdonto->idx; i++) {
                                                keys[index + i] = holdonto->keys[i];
                                                values[index + i] = holdonto->values[i];
                                                requests[index + i] = holdonto->requests[i];
                                                hashes[index + i] = holdonto->hashes[i];
                                            }
                                            index += holdonto->idx;
                                            holdonto = nullptr;
                                        }

                                        int attempts = 0;

                                        while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                                            if (this->gpu_qs[gpu].try_pop(res)) {
                                                load--;
                                                //std::cerr << "Got a batch on handler thread " << tid << "\n";
                                                if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                                    //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                                    holdonto = res;
                                                    break;
                                                }
                                                for (int i = 0; i < res->idx; i++) {
                                                    keys[index + i] = res->keys[i];
                                                    values[index + i] = res->values[i];
                                                    assert(res->requests[i] != REQUEST_INSERT || res->values[i]->size > 0);
                                                    requests[index + i] = res->requests[i];
                                                    hashes[index + i] = res->hashes[i];
                                                }
                                                writeBack.push_back({index, res});
                                                index += res->idx;
                                                if (res->flush) {
                                                    break;
                                                }
                                            } else {
                                                attempts++;
                                            }
                                        }

                                        if (index > 0) {

                                            //std::cerr << "Batching " << tid << "\n";

                                            auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                                            cudaEvent_t start, stop;

                                            gpuErrchk(cudaEventCreate(&start));
                                            gpuErrchk(cudaEventCreate(&stop));

                                            float t;

                                            slab->moveBufferToGPU(batchData, stream);
                                            gpuErrchk(cudaEventRecord(start, stream));
                                            slab->diy_batch(batchData, ceil(index / 512.0), 512, stream);
                                            gpuErrchk(cudaEventRecord(stop, stream));
                                            slab->moveBufferToCPU(batchData, stream);
                                            gpuErrchk(cudaStreamSynchronize(stream));

                                            auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                                            gpuErrchk(cudaEventElapsedTime(&t, start, stop));
                                            gpuErrchk(cudaEventDestroy(start));
                                            gpuErrchk(cudaEventDestroy(stop));
                                            int timesGoingToCache = 0;
                                            for (auto &wb : writeBack) {

                                                int rbLoc = wb.second->resBufStart;

                                                for (int i = 0; i < wb.second->idx; ++i) {

                                                    if (wb.second->handleInCache[i]) {
                                                        timesGoingToCache++;
                                                        auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                                    *(this->model));
                                                        if (cacheRes.first->valid == 1) {
                                                            data_t *cpy = nullptr;
                                                            if (cacheRes.first->deleted == 0) {
                                                                cpy = new data_t(cacheRes.first->value->size);
                                                                memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                                                            }

                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            cacheRes.first->valid = 1;
                                                            cacheRes.first->value = values[wb.first + i];
                                                            cacheRes.first->deleted = (values[wb.first + i] ==
                                                                                       EMPTY<data_t *>::value);
                                                        }
                                                        asm volatile("":: : "memory");

                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                                    } else {
                                                        if (requests[wb.first + i] == REQUEST_REMOVE) {
                                                            wb.second->resBuf->resultValues[rbLoc +
                                                                                            i] = values[wb.first + i];
                                                        } else if (requests[wb.first + i] == REQUEST_GET) {
                                                            data_t *cpy = nullptr;
                                                            if (values[wb.first + i]) {
                                                                cpy = new data_t(values[wb.first + i]->size);
                                                                memcpy(cpy->data, values[wb.first + i]->data, cpy->size);
                                                            }
                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            wb.second->resBuf->resultValues[rbLoc + i] = nullptr;
                                                        }

                                                        asm volatile("":: : "memory");
                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                                    }
                                                }
                                                delete wb.second;
                                            }

                                            mops[tid].push_back(
                                                    {std::chrono::high_resolution_clock::now(), timestampWriteBack,
                                                     timestampStartBatch,
                                                     timestampWriteToBatch, t, index, timesGoingToCache});

                                            ops += index;
                                            //std::cerr << "Batched " << tid << "\n";

                                        }
                                    }
                                    if (stream != cudaStreamDefault) gpuErrchk(cudaStreamDestroy(stream));
                                }, i, config[i].gpu,
                                gpusToSlab[config[i].gpu], config[i].stream));

        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, data_t *>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};


template<typename K, typename V, typename M>
class KVStore {
public:

    KVStore() : cache(std::make_shared<typename Cache<K, V>::type>()), model(new M()) {
        slab = std::make_shared<Slabs<K, V, M>>(STANDARD_CONFIG, this->cache, model);
    }

    KVStore(const std::vector<PartitionedSlabUnifiedConfig> &conf) : cache(
            std::make_shared<typename Cache<K, V>::type>()), model(new M()) {
        slab = std::make_shared<Slabs<K, V, M>>(conf, this->cache, model);
    }

    KVStore(const KVStore<K, V, M> &other) : slab(other.slab), cache(other.cache), model(other.model) {

    }

    ~KVStore() {

    }

    std::shared_ptr<Slabs<K, V, M>> getSlab() {
        return slab;
    }

    std::shared_ptr<typename Cache<K, V>::type> getCache() {
        return cache;
    }

    std::shared_ptr<M> getModel() {
        return model;
    }


private:
    std::shared_ptr<Slabs<K, V, M>> slab;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::shared_ptr<M> model;
};

#endif //KVGPU_KVSTORE_CUH
