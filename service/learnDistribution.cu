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

#include <zipf.hh>
#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <iostream>

constexpr float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

constexpr float d_sigmoid(float x) {
    return sigmoid(x)*(1 - sigmoid(x));
}

constexpr float serr(float y, float ypred) {
    float diff = (y - ypred);
    return diff * diff;
}

constexpr float d_serr(float y, float ypred) {
    float diff = (y - ypred);
    return 2.0f * diff;
}

int main() {

    double theta = 0.99;
    unsigned range = 10000000;
    auto zetaN = betterstd::zeta(theta, range);

    float w = rand() / (float)RAND_MAX;
    float b = 0.0;
    float alpha = 0.1f;

    for(int repeat = 0; repeat < 1000; repeat++) {
        std::vector<int> v;
        std::unordered_map<int, unsigned> m;

        for (int i = 0; i < 512; i++) {
            int gen = betterstd::rand_zipf(range, zetaN, theta);
            if (m.find(gen) != m.end()) {
                m[gen] += 1;
            } else {
                m[gen] = 1;
            }
        }

        std::unordered_map<int, float> pred;
        std::vector<int> input;

        for (auto &p : m) {
            pred[p.first] = p.second / 512.0f;
            input.push_back(p.first);
        }


        std::vector<float> output;
        std::vector<std::pair<float, float>> d_output;

        float sse = 0.0f;

        float d_w = 0.0f;
        float d_b = 0.0f;
        float sumdiff = 0.0f;

        for (auto &x : input) {
            output.push_back(sigmoid(x * w + b));

            auto d_s = d_sigmoid(x * w + b);

            auto p_d_w = x * d_s;
            auto p_d_b = d_s;

            output.push_back(sigmoid(x * w + b));
            d_output.push_back({p_d_w, p_d_b});
        }

        for (int i = 0; i < output.size(); ++i) {
            sumdiff += pred[i] - output[i];
            sse += serr(output[i], pred[i]);
            d_w += d_serr(output[i], pred[i]) * d_output[i].first;
            d_b += d_serr(output[i], pred[i]) * d_output[i].second;
        }

        std::cerr << alpha * d_w << " " << sumdiff << std::endl;
        std::cerr << alpha * d_b << " " << sumdiff << std::endl;

        w -= alpha * d_w;
        b -= alpha * d_b;

        std::cout << sse << std::endl;
    }
    return 0;
}