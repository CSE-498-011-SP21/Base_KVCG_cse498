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

#include <iostream>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <kvcg.cuh>
#include <groupallocator>
#include <set>

#ifndef KVGPU_HELPER_CUH
#define KVGPU_HELPER_CUH

struct barrier_t {
    std::condition_variable cond;
    std::mutex mtx;
    int count;
    int crossing;

    barrier_t(int n) : count(n), crossing(0) {}

    void wait() {
        std::unique_lock<std::mutex> ulock(mtx);
        /* One more thread through */
        crossing++;
        /* If not all here, wait */
        if (crossing < count) {
            cond.wait(ulock);
        } else {
            cond.notify_all();
            /* Reset for next time */
            crossing = 0;
        }
    }
};

std::vector<char> toBase256(unsigned x) {

    std::vector<char> cvec;

    while (x != 0) {
        unsigned mod = x % 256;
        x = x / 256;
        cvec.insert(cvec.begin(), mod);
    }
    return cvec;
}

data_t *unsignedToData_t(unsigned x, size_t s) {
    using namespace groupallocator;
    Context ctx;
    auto v = toBase256(x);
    data_t *d;
    allocate(&d, sizeof(data_t), ctx);
    char *underlyingData;
    allocate(&underlyingData, sizeof(char) * s, ctx);

    d->size = s;
    d->data = underlyingData;

    int k = 0;
    for (; k < v.size(); ++k) {
        underlyingData[k] = v[k];
    }
    for (; k < s; ++k) {
        underlyingData[k] = '\0';
    }
    return d;
}
#endif //KVGPU_HELPER_CUH
