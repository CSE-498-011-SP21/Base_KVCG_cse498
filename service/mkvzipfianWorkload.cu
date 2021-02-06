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
#include <Request.cuh>
#include <vector>
#include <zipf.hh>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace pt = boost::property_tree;

double zetaN = 0.1;

struct ZipfianWorkloadConfig {
    ZipfianWorkloadConfig() {
        theta = 0.99;
        range = 100000000;
        //n = 1000000;
        keysize = 8;
        ratio = 100;
    }

    ZipfianWorkloadConfig(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        theta = root.get<double>("theta");
        range = root.get<int>("range");
        //n = root.get<int>("n");
        keysize = root.get<size_t>("keysize");
        ratio = root.get<int>("ratio");
    }


    ~ZipfianWorkloadConfig() {}

    double theta;
    int range;
    //int n;
    size_t keysize;
    int ratio;
};

ZipfianWorkloadConfig zipfianWorkloadConfig;

extern "C" void initWorkload() {
    zetaN = betterstd::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

extern "C" void initWorkloadFile(std::string filename) {
    zipfianWorkloadConfig = ZipfianWorkloadConfig(filename);
    zetaN = betterstd::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

std::shared_ptr<megakv::BatchOfRequests>
generateWorkloadZipfLargeKey(size_t keySize, unsigned size, double theta, int n, double zetaN, unsigned *seed,
                             int ratioOfReads) {

    std::shared_ptr<megakv::BatchOfRequests> batch = std::make_shared<megakv::BatchOfRequests>();

    std::string longValue;
    for (size_t i = 0; i < keySize; i++) {
        longValue += 'a';
    }

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            batch->reqs[i].key = std::to_string(betterstd::rand_zipf_r(seed, n, zetaN, theta));
            batch->reqs[i].value = "";
            batch->reqs[i].requestInt = megakv::REQUEST_GET;
        } else {
            if (rand_r(seed) % 100 < 50) {
                batch->reqs[i].key = std::to_string(betterstd::rand_zipf_r(seed, n, zetaN, theta));
                batch->reqs[i].value = longValue;
                batch->reqs[i].requestInt = megakv::REQUEST_INSERT;
            } else {
                batch->reqs[i].key = std::to_string(betterstd::rand_zipf_r(seed, n, zetaN, theta));
                batch->reqs[i].value = "";
                batch->reqs[i].requestInt = megakv::REQUEST_REMOVE;
            }
        }

    }
    return batch;

}

extern "C" std::shared_ptr<megakv::BatchOfRequests> generateWorkloadBatch(unsigned int *seed, unsigned batchsize) {
    return generateWorkloadZipfLargeKey(zipfianWorkloadConfig.keysize, batchsize, zipfianWorkloadConfig.theta,
                                        zipfianWorkloadConfig.range, zetaN, seed, zipfianWorkloadConfig.ratio);
}
