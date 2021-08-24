#ifndef AITM_H_
#define AITM_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

struct Tower : public torch::nn::Module {
  Tower(int64_t input_dim, std::vector<int64_t> dims,
        std::vector<double> drop_prob) {
    layer_ = register_module(
        "layer", torch::nn::Sequential(
                     torch::nn::Linear(input_dim, dims[0]), torch::nn::ReLU(),
                     torch::nn::Dropout(drop_prob[0]),
                     torch::nn::Linear(dims[0], dims[1]), torch::nn::ReLU(),
                     torch::nn::Dropout(drop_prob[1]),
                     torch::nn::Linear(dims[1], dims[2]), torch::nn::ReLU(),
                     torch::nn::Dropout(drop_prob[2])));
  }
  torch::Tensor forward(torch::Tensor input) {
    auto x = input.flatten(-1);
    return layer_->forward(x);
  }
  torch::nn::Sequential layer_{nullptr};
};

struct Attention : public torch::nn::Module {
  Attention(int64_t dim) : dim_{dim} {
    q_layer_ = register_module(
        "q_layer",
        torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
    k_layer_ = register_module(
        "k_layer",
        torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
    v_layer_ = register_module(
        "v_layer",
        torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  }

  torch::Tensor forward(torch::Tensor input) {
    auto Q = q_layer_->forward(input);
    auto K = k_layer_->forward(input);
    auto V = v_layer_->forward(input);

    auto a = at::sum(at::mul(Q, V), -1) / at::sqrt(torch::tensor(dim_));
    auto as = softmax_->forward(a);
    auto out = at::sum(at::mul(as.unsqueeze(-1), V), 1);
    return out;
  }
  torch::nn::Linear q_layer_{nullptr};
  torch::nn::Linear k_layer_{nullptr};
  torch::nn::Linear v_layer_{nullptr};
  torch::nn::Softmax softmax_{torch::nn::SoftmaxOptions(1)};
  int64_t dim_;
};

struct AITM : public torch::nn::Module {
  AITM(std::unordered_map<std::string, int64_t> feature_vocabulary,
       int64_t embedding_size) {
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> emb_list;
    for (auto item : feature_vocabulary) {
      feature_names_.push_back(item.first);
      emb_list.push_back(std::pair<std::string, std::shared_ptr<Module>>(
          item.first, torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                               item.second, embedding_size))
                          .ptr()));
    }
    std::sort(feature_names_.begin(), feature_names_.end());
    embedding_dict_ =
        register_module("embedding_dict", torch::nn::ModuleDict(emb_list));

    int64_t tower_input_size = feature_vocabulary.size() * embedding_size;
    std::vector<int64_t> tower_dims = {128, 64, 32};
    std::vector<double> drop_prob = {0.1, 0.3, 0.3};

    click_tower_ = register_module(
        "click_tower",
        std::make_shared<Tower>(tower_input_size, tower_dims, drop_prob));
    conversion_tower_ = register_module(
        "conversion_tower",
        std::make_shared<Tower>(tower_input_size, tower_dims, drop_prob));
    attention_layer_ =
        register_module("attention_layer", std::make_shared<Attention>(32));
    info_layer_ = register_module(
        "info_layer",
        torch::nn::Sequential(torch::nn::Linear(32, 32), torch::nn::ReLU(),
                              torch::nn::Dropout(drop_prob[2])));
    click_layer_ = register_module(
        "click_layer",
        torch::nn::Sequential(torch::nn::Linear(32, 1), torch::nn::Sigmoid()));
    conversion_layer_ = register_module(
        "conversion_layer",
        torch::nn::Sequential(torch::nn::Linear(32, 1), torch::nn::Sigmoid()));
  }

  std::pair<torch::Tensor, torch::Tensor>
  forward(std::unordered_map<std::string, torch::Tensor> inputs) {
    std::vector<torch::Tensor> emb_list;
    for (const auto &name : feature_names_) {
      auto emb = embedding_dict_[name]->as<torch::nn::Embedding>()->forward(
          inputs[name]);
      emb_list.push_back(emb);
    }
    auto feature_embedding = at::cat(emb_list, 1);
    auto tower_click = click_tower_->forward(feature_embedding);
    auto tower_conversion =
        conversion_tower_->forward(feature_embedding).unsqueeze(1);
    auto info = info_layer_->forward(tower_click).unsqueeze(1);
    auto ait = attention_layer_->forward(at::cat({tower_conversion, info}, 1));
    auto click = click_layer_->forward(tower_click).squeeze(1);
    auto conversion = conversion_layer_->forward(ait).squeeze(1);
    return std::pair<torch::Tensor, torch::Tensor>{click, conversion};
  }

  std::shared_ptr<Tower> click_tower_{nullptr};
  std::shared_ptr<Tower> conversion_tower_{nullptr};
  std::shared_ptr<Attention> attention_layer_{nullptr};
  torch::nn::Sequential info_layer_{nullptr};
  torch::nn::ModuleDict embedding_dict_{nullptr};
  torch::nn::Sequential click_layer_{nullptr};
  torch::nn::Sequential conversion_layer_{nullptr};
  std::vector<std::string> feature_names_;
};

void loadAITM(std::shared_ptr<AITM> model, const std::string &model_file) {
  auto tm = torch::jit::load(model_file);
  auto model_parameters = model->named_parameters();
  for (auto param : tm.named_parameters()) {
    auto name = param.name;
    auto dist = model_parameters.find(name);
    if (dist == nullptr) {
      std::cout << "parameter not match name: " << name << "\n";
    } else {
      dist->copy_(param.value);
    }
  }
}

#endif
