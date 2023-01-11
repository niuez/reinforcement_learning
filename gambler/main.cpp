#include <random>
#include <vector>
#include <iostream>
#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;

struct Coin {
  double p;
};

void solve(const Coin& coin) {
  constexpr int MAX = 100;
  constexpr int Iteration = 10000;
  constexpr int Gamma = 1;

  std::vector<double> V(MAX + 1, 0);

  auto calc_sigma = [&](int s, int a) {
    double sigma = 0;
    {
      // win
      int n = std::min(MAX, s + a);
      double r = n == MAX ? 1 : 0;
      sigma += coin.p * (r + Gamma * V[n]);
    }
    {
      // lose
      int n = s - a;
      double r = 0;
      sigma += (1 - coin.p) * (r + Gamma * V[n]);
    }
    return sigma;
  };

  for(int q = 0; q < Iteration; q++) {
    double delta = 0;
    for(int s = 1; s < MAX; s++) {
      double before = V[s];
      V[s] = std::numeric_limits<double>::min();
      for(int a = 1; a <= s; a++) {
        V[s] = std::max(V[s], calc_sigma(s, a));
      }
      delta = std::max(delta, std::abs(before - V[s]));
    }
    if(q % 100 == 0) {
      std::cerr << q << "=" << delta << std::endl;
    }
  }

  std::vector<double> x;
  std::vector<double> vs;
  std::vector<double> ps;

  std::vector<double> cx;
  std::vector<double> cy;
  for(int s = 1; s < MAX; s++) {
    int pi = 0;
    double v = std::numeric_limits<double>::min();
    std::vector<int> cand;
    for(int a = 1; a <= s; a++) {
      double sigma = calc_sigma(s, a);
      if(std::abs(v - sigma) < 1e-9) {
        cand.push_back(a);
      }
      else if(v < sigma) {
        v = sigma;
        pi = a;
        cand.clear();
        cand.push_back(a);
      }
    }
    x.push_back(s);
    vs.push_back(v);
    ps.push_back(pi);

    for(auto v: cand) {
      cx.push_back(s);
      cy.push_back(v);
    }
  }
  plt::figure_size(1200, 720);
  plt::title("p_h = " + std::to_string(coin.p));
  plt::subplot(2, 1, 1);
  plt::plot(x, vs);
  plt::subplot(2, 1, 2);
  plt::scatter(cx, cy);
  plt::save("p" + std::to_string(coin.p) + ".png");
  plt::clf();
}

int main() {
  Coin coin;
  coin.p = 0.4;
  solve(coin);
  coin.p = 0.25;
  solve(coin);
  coin.p = 0.55;
  solve(coin);
}


