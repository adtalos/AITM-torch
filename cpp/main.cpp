#include "aitm.h"

#include <chrono>
#include <iostream>
#include <torch/script.h>

int main() {
  std::unordered_map<std::string, torch::Tensor> inputs;
  // 101 tensor([56460, 56460])
  inputs.insert({"101", torch::tensor({56460, 56460})});
  inputs.insert({"121", torch::tensor({15, 15})});
  inputs.insert({"122", torch::tensor({4, 4})});
  inputs.insert({"124", torch::tensor({1, 1})});
  inputs.insert({"125", torch::tensor({4, 4})});
  inputs.insert({"126", torch::tensor({0, 0})});
  inputs.insert({"127", torch::tensor({2, 2})});
  inputs.insert({"128", torch::tensor({2, 2})});
  inputs.insert({"129", torch::tensor({2, 2})});
  inputs.insert({"205", torch::tensor({303798, 25006})});
  inputs.insert({"206", torch::tensor({38, 5973})});
  inputs.insert({"207", torch::tensor({133408, 101467})});
  inputs.insert({"216", torch::tensor({23965, 51505})});
  inputs.insert({"508", torch::tensor({2342, 1493})});
  inputs.insert({"509", torch::tensor({0, 0})});
  inputs.insert({"702", torch::tensor({0, 0})});
  inputs.insert({"853", torch::tensor({2707, 4062})});
  inputs.insert({"301", torch::tensor({3, 3})});

  std::unordered_map<std::string, int64_t> feature_vocabulary = {
      {"101", 238635}, {"121", 98},     {"122", 14},     {"124", 3},
      {"125", 8},      {"126", 4},      {"127", 4},      {"128", 3},
      {"129", 5},      {"205", 467298}, {"206", 6929},   {"207", 263942},
      {"216", 106399}, {"508", 5888},   {"509", 104830}, {"702", 51878},
      {"853", 37148},  {"301", 4},
  };

  std::string modelfile ="../../python/out/AITM.model.pt";
  torch::NoGradGuard no_grad;
  std::shared_ptr<AITM> model = std::make_shared<AITM>(feature_vocabulary, 5);
  loadAITM(model, modelfile);
  model->eval();
  auto s = model->forward(inputs);
  std::cout << s.first << " second: " << s.second << "\n";

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++) {
    auto res = model->forward(inputs);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Predict Cost: " << duration.count() << " microseconds"
            << std::endl;
  std::cout << "finish \n";
  return 0;
}
