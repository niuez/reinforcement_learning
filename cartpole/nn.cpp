#include <vector>
#include <cmath>
#include <iostream>
#include "cartpole.hpp"

struct NN {
  using value_type = double;
  std::vector<std::vector<value_type>> value;
  std::vector<std::vector<value_type>> theta;

  template<class URGB>
  NN(URGB& g) {
    value.resize(5);
    theta.resize(4);

    value[0].resize(4);
    value[1].resize(32);
    value[2].resize(32);
    value[3].resize(2);
    value[4].resize(2);

    theta[0].resize(4 * 32 + 32);
    theta[2].resize(32 * 2 + 2);

    for(int i = 0; i < theta.size(); i++) {
      for(int j = 0; j < theta[i].size(); j++) {
        if((i == 0 && j % 5 == 4) || (i == 2 && j % 33 == 32)) {
          theta[i][j] = std::uniform_real_distribution<value_type>(- 1.0 / value[i].size(), 1.0 / value[i].size())(g);
        }
        else {
          theta[i][j] = std::normal_distribution<value_type>(0, std::sqrt(2.0 / value[i].size()))(g);
        }
      }
    }
  }

  void evaluate() {
    {
      for(int i = 0; i < value[1].size(); i++) {
        value[1][i] = 0;
        for(int j = 0; j < value[0].size(); j++) {
          value[1][i] += value[0][j] * theta[0][i * (value[0].size() + 1) + j];
        }
        value[1][i] += theta[0][i * (value[0].size() + 1) + value[0].size()];
      }
    }
    {
      for(int i = 0; i < value[1].size(); i++) {
        value[2][i] = std::max(value[1][i], value_type(0));
      }
    }
    {
      for(int i = 0; i < value[3].size(); i++) {
        value[3][i] = 0;
        for(int j = 0; j < value[2].size(); j++) {
          value[3][i] += value[2][j] * theta[2][i * (value[2].size() + 1) + j];
        }
        value[3][i] += theta[2][i * (value[2].size() + 1) + value[2].size()];
      }
    }
    {
      int min_i = std::min_element(value[3].begin(), value[3].end()) - value[3].begin();
      if(value[3][1 - min_i] - value[3][min_i] >= 30) {
        value[4][min_i] = 0;
        value[4][1 - min_i] = 1;
      }
      else {
        float sigma = 0;
        for(int i = 0; i < value[3].size(); i++) {
          sigma += std::exp(value[3][i] - value[3][min_i]);
        }
        for(int i = 0; i < value[3].size(); i++) {
          value[4][i] = std::exp(value[3][i] - value[3][min_i]) / sigma;
        }
      }
    }
  }

  std::vector<std::vector<value_type>> log_gradient(int out_i) const {
    std::vector<std::vector<value_type>> g_va = value;
    std::vector<std::vector<value_type>> g_th = theta;

    g_va[4][out_i] = 1;

    {
      for(int i = 0; i < g_va[3].size(); i++) {
        if(i == out_i) {
          g_va[3][i] = (1 - value[4][i]);
        }
        else {
          g_va[3][i] = -value[4][i];
        }
      }
    }
    {
      for(int j = 0; j < value[2].size(); j++) {
        g_va[2][j] = 0;
      }
      for(int i = 0; i < value[3].size(); i++) {
        for(int j = 0; j < value[2].size(); j++) {
          g_th[2][i * (value[2].size() + 1) + j] = g_va[3][i] * value[2][j];
          g_va[2][j] += g_va[3][i] * theta[2][i * (value[2].size() + 1) + j];
        }
        g_th[2][i * (value[2].size() + 1) + value[2].size()] = g_va[3][i];
      }
    }
    {
      for(int i = 0; i < value[1].size(); i++) {
        g_va[1][i] = g_va[2][i] * (value[1][i] <= 0 ? 0 : 1);
      }
    }
    {
      for(int j = 0; j < value[0].size(); j++) {
        g_va[0][j] = 0;
      }
      for(int i = 0; i < value[1].size(); i++) {
        for(int j = 0; j < value[0].size(); j++) {
          g_th[0][i * (value[0].size() + 1) + j] = g_va[1][i] * value[0][j];
          g_va[0][j] += g_va[1][i] * theta[0][i * (value[0].size() + 1) + j];
        }
        g_th[0][i * (value[0].size() + 1) + value[0].size()] = g_va[1][i];
      }
    }

    return g_th;
  }

  std::vector<std::vector<value_type>> mt;
  std::vector<std::vector<value_type>> vt;
  double bp1 = 1;
  double bp2 = 1;

  void sgd(const std::vector<std::vector<value_type>>& th, const value_type& G, const value_type lr) {
    for(int i = 0; i < th.size(); i++) {
      for(int j = 0; j < th[i].size(); j++) {
        theta[i][j] -= lr * th[i][j] * G;
      }
    }
  }
  void adam(const std::vector<std::vector<value_type>>& th, const value_type& G, const value_type lr) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    if(mt.empty()) {
      mt = th;
      vt = th;
      bp1 = 1;
      bp2 = 1;
      for(int i = 0; i < th.size(); i++) {
        for(int j = 0; j < th[i].size(); j++) {
          mt[i][j] = vt[i][j] = 0;
        }
      }
    }
    bp1 *= beta1;
    bp2 *= beta2;

    for(int i = 0; i < th.size(); i++) {
      for(int j = 0; j < th[i].size(); j++) {
        mt[i][j] = beta1 * mt[i][j] + (1 - beta1) * th[i][j] * G;
        vt[i][j] = beta2 * vt[i][j] + (1 - beta2) * std::pow(th[i][j] * G, 2);
        theta[i][j] -= lr * mt[i][j] / (1 - bp1) / (std::sqrt(vt[i][j] / (1 - bp2)) + 1e-9);
      }
    }
  }
};

int main() {
  constexpr int MAX_TRY = 500;
  constexpr int MAX_STEP = 200;
  constexpr double Gamma = 0.99;
  using value_type = NN::value_type;
  CartPole_v1 state;
  std::mt19937 mt(788);

  NN nn(mt);

  int ok = 0;

  for(int ti = 0; ti < MAX_TRY; ti++) {
    state.reset(mt);
    int t = 0;
    std::vector<value_type> rs;
    std::vector<std::vector<std::vector<value_type>>> gs;
    for(; t < MAX_STEP; t++) {
      nn.value[0][0] = state.state.cart_pos;
      nn.value[0][1] = state.state.cart_vel;
      nn.value[0][2] = state.state.pole_pos;
      nn.value[0][3] = state.state.pole_vel;
      nn.evaluate();
      int action = std::bernoulli_distribution(nn.value[4][0])(mt) ? 0 : 1;
      //std::cerr << state.state << " " << nn.value[4][0] << " " << nn.value[4][1] << " " << action << std::endl;
      gs.push_back(nn.log_gradient(action));
      bool is_saved = state.step(action) && (t + 1) < MAX_STEP;
      float reward = 0;
      if(!is_saved) {
        reward = t < MAX_STEP - 5 ? -1 : 1;
      }
      rs.push_back(reward);
      if(!is_saved) {
        break;
      }
    }
    //std::cerr << state.state << std::endl;
    std::cerr << ti << "\t:" << t << std::endl;
    if(t + 1 == MAX_STEP) {
      ok++;
      if(ok == 10) {
        std::cerr << "10 consecutive successes" << std::endl;
        break;
      }
    }
    else {
      ok = 0;
    }

    double G = 0;
    std::vector<double> Gs(rs.size());
    double G_sum = 0;
    double G_s2 = 0;
    for(int i = rs.size(); i --> 0;) {
      G = G * Gamma + rs[i];
      Gs[i] = G;
      G_sum += G;
      G_s2 += G * G;
    }
    double mean = G_sum / rs.size();
    double stddev = std::sqrt(G_s2 / rs.size() - mean * mean);

    std::vector<std::vector<value_type>> delta;
    for(int i = rs.size(); i --> 0;) {
      Gs[i] = (Gs[i] - mean) / (stddev + 1e-9);
      if(delta.empty()) {
        delta = gs[i];
        for(int x = 0; x < gs[i].size(); x++) {
          for(int y = 0; y < gs[i][x].size(); y++) {
            delta[x][y] = gs[i][x][y] * -Gs[i];
          }
        }
      }
      else {
        for(int x = 0; x < gs[i].size(); x++) {
          for(int y = 0; y < gs[i][x].size(); y++) {
            delta[x][y] += gs[i][x][y] * -Gs[i];
          }
        }
      }
    }
    nn.adam(delta, 1, 1e-2); // t=201で200stepを連続10回出せた
  }
}
