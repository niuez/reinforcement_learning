#include "k_armed_bandit.hpp"
#include "matplotlibcpp.hpp"
#include <random>
#include <random>
#include <iostream>
#include <numeric>

namespace plt = matplotlibcpp;

template<class URGB>
MultiBandit init_bandits(URGB& g) {
  constexpr int N = 10;
  constexpr double bandit_gen_mean = 0;
  constexpr double bandit_gen_stddev = 1;
  constexpr double bandit_stddev = 1;
  std::normal_distribution<double> bandit_gen(bandit_gen_mean, bandit_gen_stddev);
  MultiBandit bandits(g, N, bandit_gen, bandit_stddev);
  std::cerr << bandits << std::endl;
  return bandits;
}

template<class URGB>
void eps_greedy(std::string name, URGB& g, MultiBandit& bandits, const double eps, const int steps, double Q_init = 0) {
  std::vector<double> cnt(bandits.size(), 0);
  std::vector<double> Q(bandits.size(), Q_init);
  std::bernoulli_distribution select_dist(eps);
  std::uniform_int_distribution<int> action_dist(0, bandits.size() - 1);
  std::vector<double> step;
  std::vector<double> aves;
  std::vector<double> best_cnt;
  double sum = 0;
  double best = 0;
  for(int s = 0; s < steps; s++) {
    int action = -1;
    if(select_dist(g)) {
      // random
      action = action_dist(g);
    }
    else {
      // greedy
      action = std::max_element(Q.begin(), Q.end()) - Q.begin();
    }
    double R = bandits.reward(g, action);
    cnt[action]++;
    Q[action] += (R - Q[action]) / cnt[action];

    sum += R;
    best += action == bandits.best_action() ? 1 : 0;
    step.push_back(s + 1);
    aves.push_back(sum / (s + 1));
    best_cnt.push_back(best / (s + 1));
  }
  for(int i = 0; i < bandits.size(); i++) {
    std::cerr << "#" << i << ": Q=" << Q[i] << std::endl;
  }

  plt::subplot(1, 2, 1);
  plt::named_plot(name, step, aves);
  plt::subplot(1, 2, 2);
  plt::named_plot(name, step, best_cnt);
}

template<class URGB>
void upper_confidence_bound(std::string name, URGB& g, MultiBandit& bandits, const int steps, double c, double Q_init = 0) {
  std::vector<double> cnt(bandits.size(), 0);
  std::vector<double> Q(bandits.size(), Q_init);

  std::vector<double> step;
  std::vector<double> aves;
  std::vector<double> best_cnt;
  double sum = 0;
  double best = 0;
  for(int s = 0; s < steps; s++) {
    int action = 0;
    double MAX = std::numeric_limits<double>::min();
    for(int i = 0; i < bandits.size(); i++) {
      double ucb = cnt[i] == 0 ? std::numeric_limits<double>::max() : (Q[i] + c * std::sqrt(std::log(s + 1) / cnt[i]));
      if(MAX < ucb) {
        MAX = ucb;
        action = i;
      }
    }
    double R = bandits.reward(g, action);
    cnt[action]++;
    Q[action] += (R - Q[action]) / cnt[action];

    sum += R;
    best += action == bandits.best_action() ? 1 : 0;
    step.push_back(s + 1);
    aves.push_back(sum / (s + 1));
    best_cnt.push_back(best / (s + 1));
  }
  for(int i = 0; i < bandits.size(); i++) {
    std::cerr << "#" << i << ": Q=" << Q[i] << std::endl;
  }

  plt::subplot(1, 2, 1);
  plt::named_plot(name, step, aves);
  plt::subplot(1, 2, 2);
  plt::named_plot(name, step, best_cnt);
}

template<class URGB>
void gradient_ascent(std::string name, URGB& g, MultiBandit& bandits, const int steps, double alpha) {
  std::vector<double> H(bandits.size(), 0);
  std::vector<double> pi(bandits.size(), 1.0 / bandits.size());
  std::uniform_real_distribution<double> pi_dist(0, 1);

  std::vector<double> step;
  std::vector<double> aves;
  std::vector<double> best_cnt;
  double sum = 0;
  double best = 0;
  for(int s = 0; s < steps; s++) {
    int action = 0;
    double dice = pi_dist(g);
    while(dice >= pi[action]) {
      dice -= pi[action];
      action++;
    }
    double R = bandits.reward(g, action);
    double R_bar = (sum + R) / (s + 1);
    double sigma = 0;
    for(int i = 0; i < bandits.size(); i++) {
      if(i == action) {
        H[i] += alpha * (R - R_bar) * (1 - pi[i]);
      }
      else {
        H[i] -= alpha * (R - R_bar) * pi[i];
      }
      sigma += std::exp(H[i]);
    }
    for(int i = 0; i < bandits.size(); i++) {
      pi[i] = std::exp(H[i]) / sigma;
    }

    sum += R;
    best += action == bandits.best_action() ? 1 : 0;
    step.push_back(s + 1);
    aves.push_back(sum / (s + 1));
    best_cnt.push_back(best / (s + 1));
  }
  for(int i = 0; i < bandits.size(); i++) {
    std::cerr << "#" << i << ": pi=" << pi[i] << std::endl;
  }

  plt::subplot(1, 2, 1);
  plt::named_plot(name, step, aves);
  plt::subplot(1, 2, 2);
  plt::named_plot(name, step, best_cnt);
}

int main() {
  constexpr int seed = 768;
  std::mt19937 mt(seed);
  auto bandits = init_bandits(mt);

  plt::figure_size(1000, 500);
  plt::suptitle("10 armed bandit");

  eps_greedy("eps0.1", mt, bandits, 0.1, 5000);
  eps_greedy("eps0.01", mt, bandits, 0.01, 5000);
  eps_greedy("eps0.1+optmistic=5", mt, bandits, 0.1, 5000, 5);
  upper_confidence_bound("ucb c=2", mt, bandits, 5000, 2, 0);
  gradient_ascent("gradient ascent alpha=0.1", mt, bandits, 5000, 0.1);
  
  plt::subplot(1, 2, 1);
  plt::legend();
  plt::subplot(1, 2, 2);
  plt::legend();

  plt::save("k_armed_bandit.png");
}
