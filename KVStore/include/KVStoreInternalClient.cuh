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

#include "KVStore.cuh"
#include <functional>
#include <chrono>
#include <tbb/concurrent_queue.h>

#ifndef KVGPU_KVSTOREINTERNALCLIENT_CUH
#define KVGPU_KVSTOREINTERNALCLIENT_CUH

int LOAD_THRESHOLD = BLOCKS * 10000;

template<typename K, typename V>
struct RequestWrapper {
    K key;
    V value;
    //std::shared_ptr<SharedResult<std::pair<bool, V>>> getPromise;
    //std::shared_ptr<SharedResult<bool>> otherPromise;
    unsigned requestInteger;
};

struct block_t {
    std::condition_variable cond;
    std::mutex mtx;
    int count;
    int crossing;

    block_t(int n) : count(n), crossing(0) {}

    void wait() {
        std::unique_lock<std::mutex> ulock(mtx);

        crossing++;

        // wait here

        cond.wait(ulock);
    }

    bool threads_blocked() {
        std::unique_lock<std::mutex> ulock(mtx);
        return crossing == count;
    }

    void wake() {
        std::unique_lock<std::mutex> ulock(mtx);
        cond.notify_all();
        crossing = 0;
    }

};

template<typename K, typename V>
void schedule_for_batch_helper(K *&keys, V *&values, unsigned *requests, unsigned *hashes,
                               std::unique_lock<kvgpu::mutex> *locks, unsigned *&correspondence,
                               int &index, const int &hash, const RequestWrapper<K, V> &req,
                               std::unique_lock<kvgpu::mutex> &&lock, int i) {

    keys[index] = req.key;
    values[index] = req.value;
    requests[index] = req.requestInteger;
    hashes[index] = hash;
    locks[index] = std::move(lock);
    correspondence[index] = i;
    index++;
}

/**
 * K is the type of the Key
 * V is the type of the Value
 * M is the type of the Model
 * @tparam K
 * @tparam V
 * @tparam M
 */
template<typename K, typename V, typename M>
class KVStoreInternalClient {
public:
    KVStoreInternalClient(std::shared_ptr<Slabs<K, V, M>> s, std::shared_ptr<typename Cache<K, V>::type> c,
                          std::shared_ptr<M> m) : numslabs(s->numslabs),
                                                  slabs(s), cache(c), hits(0),
                                                  operations(0), start(std::chrono::high_resolution_clock::now()),
                                                  model(m) {

    }

    ~KVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    /**
     * Performs the batch of operations given
     * @param req_vector
     */
    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        bool dontDoGPU = false;

        if (slabs->load >= LOAD_THRESHOLD) {
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (model->operator()(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = 0;//cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches; //0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key,
                            cache_batch_idx.second,
                            *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }

                    if (pair.first != nullptr)
                        pair.second.unlock();
                    times.push_back(std::chrono::high_resolution_clock::now());
                }


            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches2[i]->idx > 0) {
                    gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                    sizeForGPUBatches += gpu_batches2[i]->idx;
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches2[i]);
                } else {
                    delete gpu_batches2[i];
                }
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches2[i];
            }
            resBuf->retryGPU = true;
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<V> *log_values = cache->log_values;

        tbb::concurrent_vector<int> *tmp_log_requests = new tbb::concurrent_vector<int>(
                cache->getN() * cache->getSETS());
        tbb::concurrent_vector<unsigned> *tmp_log_hash = new tbb::concurrent_vector<unsigned>(
                cache->getN() * cache->getSETS());
        tbb::concurrent_vector<K> *tmp_log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        tbb::concurrent_vector<V> *tmp_log_values = new tbb::concurrent_vector<V>(cache->getN() * cache->getSETS());

        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = tmp_log_requests;
        cache->log_hash = tmp_log_hash;
        cache->log_keys = tmp_log_keys;
        cache->log_values = tmp_log_values;
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        //std::cerr << "Tmp size " << tmpSize << "\n";

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            for (int i = 0; i < numslabs; ++i) {
                gpu_batches[i] = new BatchData<K, V>(0, std::make_shared<ResultsBuffers<V>>(batchSizeUsed),
                                                     batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }
        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    M getModel() {
        return *model;
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slabs<K, V, M>> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};

template<typename K, typename M>
class KVStoreInternalClient<K, data_t, M> {
public:
    KVStoreInternalClient(std::shared_ptr<Slabs<K, data_t *, M>> s,
                          std::shared_ptr<typename Cache<K, data_t *>::type> c, std::shared_ptr<M> m) : numslabs(
            s->numslabs), slabs(s), cache(c), hits(0),
                                                                                                        operations(0),
                                                                                                        start(std::chrono::high_resolution_clock::now()),
                                                                                                        model(m) {

    }

    ~KVStoreInternalClient() {}

    typedef RequestWrapper<K, data_t *> RW;

    /**
     * Performs the batch of operations given
     * @param req_vector
     */
    void batch(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        bool dontDoGPU = false;

        if (slabs->load >= LOAD_THRESHOLD) {
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        //req_vector.size() % 512 == 0 &&
        assert(req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, data_t> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, data_t>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (model->operator()(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
        }

        auto gpu_batches2 = std::vector<BatchData<K, data_t> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, data_t>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, data_t *> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";
                        data_t *cpy = nullptr;
                        if (pair.first->deleted == 0 && pair.first->value) {
                            cpy = new data_t(pair.first->value->size);
                            memcpy(cpy->data, pair.first->value->data, cpy->size);
                        }
                        resBuf->resultValues[responseLocationInResBuf] = cpy;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());
                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, data_t *> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            if (pair.first->valid == 1) {
                                resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                                pair.first->value = nullptr;
                            }

                            pair.first->deleted = 1;
                            pair.first->valid = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches2[i]->idx > 0) {
                    gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                    sizeForGPUBatches += gpu_batches2[i]->idx;
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches2[i]);
                } else {
                    delete gpu_batches2[i];
                }
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches2[i];
            }
            resBuf->retryGPU = true;
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<data_t *> *log_values = cache->log_values;

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = new tbb::concurrent_vector<int>(cache->getN() * cache->getSETS());
        cache->log_hash = new tbb::concurrent_vector<unsigned>(cache->getN() * cache->getSETS());
        cache->log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        cache->log_values = new tbb::concurrent_vector<data_t *>(cache->getN() * cache->getSETS());
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            auto gpu_batches = std::vector<BatchData<K, data_t *> *>(numslabs);

            for (int i = 0; i < numslabs; ++i) {
                gpu_batches[i] = new BatchData<K, data_t *>(0,
                                                            std::make_shared<ResultsBuffers<data_t *>>(batchSizeUsed),
                                                            batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }

        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    M getModel() {
        return *model;
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slabs<K, data_t *, M>> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, data_t *>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};


template<typename K, typename V, typename M>
class NoCacheKVStoreInternalClient {
public:
    NoCacheKVStoreInternalClient(std::shared_ptr<Slabs<K, V, M>> s, std::shared_ptr<typename Cache<K, V>::type> c,
                                 std::shared_ptr<M> m) : numslabs(s->numslabs), slabs(s), cache(c),
                                                         hits(0), operations(0),
                                                         start(std::chrono::high_resolution_clock::now()), model(m) {

    }

    ~NoCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    /**
     * Performs the batch of operations given
     * @param req_vector
     */
    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (mfn(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        for (int i = 0; i < numslabs; ++i) {
            slabs->gpu_qs[i].push(gpu_batches[i]);
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = 0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::locktype> pair = cache->get(req_vector_elm.key,
                                                                                             cache_batch_idx.second,
                                                                                             *model);
                    if (pair.first == nullptr) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        auto endTime = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        for (int i = 0; i < numslabs; ++i) {
            if (gpu_batches2[i]->idx > 0) {
                gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                sizeForGPUBatches += gpu_batches2[i]->idx;
                slabs->gpu_qs[i].push(gpu_batches2[i]);
            } else {
                delete gpu_batches2[i];
            }
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slabs<K, V, M>> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
};


template<typename K, typename V, typename M>
class JustCacheKVStoreInternalClient {
public:
    JustCacheKVStoreInternalClient(std::shared_ptr<Slabs<K, V, M>> s, std::shared_ptr<typename Cache<K, V>::type> c,
                                   std::shared_ptr<M> m) : numslabs(s->numslabs), slabs(s), cache(c), hits(0),
                                                           operations(0),
                                                           start(std::chrono::high_resolution_clock::now()), model(m) {

    }

    ~JustCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    /**
     * Performs the batch of operations given
     * @param req_vector
     */
    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (mfn(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        for (int i = 0; i < numslabs; ++i) {
            delete gpu_batches[i];
            //slabs->gpu_qs[i].push(gpu_batches[i]);
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = 0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::locktype> pair = cache->get(req_vector_elm.key,
                                                                                             cache_batch_idx.second,
                                                                                             *model);
                    if (pair.first == nullptr) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        auto endTime = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        for (int i = 0; i < numslabs; ++i) {
            if (gpu_batches2[i]->idx > 0) {
                gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                sizeForGPUBatches += gpu_batches2[i]->idx;
                delete gpu_batches2[i];
                //slabs->gpu_qs[i].push(gpu_batches2[i]);
            } else {
                delete gpu_batches2[i];
            }
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slabs<K, V, M>> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
};

#endif //KVGPU_KVSTOREINTERNALCLIENT_CUH
